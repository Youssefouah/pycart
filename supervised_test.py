import base64
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import accuracy_score
import pickle


@st.cache_data
def wrangle(filepath):
    return pd.read_csv(filepath)

st.title("Projects for supervised learning")
st.header("You can use this app for exploring your data, conducting thorough analysis, and creating the best model for supervised learning.")

file =  st.file_uploader("Upload file",type  = ["csv"])
if file is not None:
    df  = wrangle(file)
    st.write("The shape of the data :" ,df.shape)
    col_show = st.multiselect("Select column to show :",df.columns.tolist(),default= df.columns.to_list())
    num = st.slider("Choose number of rows :" ,min_value=5,max_value=len(df),step=1)
    st.write(df[:num][col_show])
    data = df.copy()
    df = df[:num] 

    st.header("exploratory the data")
    tab1, tab2  = st.tabs(["scatter plot","histogram"])
    col_num = df.select_dtypes(include = np.number).columns.to_list()
    with tab1:
        col1,col2,col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("select  column on x axis :",col_num )
        with col2:   
            y_col = st.selectbox("select  column on y axis :",col_num )
        with col3:    
            color  = st.selectbox("select  column to be color:",df.columns)

        fig_scatter = px.scatter(df,x=x_col,y=y_col,color=color)
        st.plotly_chart(fig_scatter)
    
    with tab2:
        feature_hist = st.selectbox("select  feature to histogram :",col_num )
        fig_histogram = px.histogram(df,x=feature_hist)
        st.plotly_chart(fig_histogram)



  # select model and use paycart
    st.header("Model ")
    
    target1 = st.selectbox("Select the target of your model:",data.columns,index=None)
    if (target1 is not None):
        if len(data[target1].value_counts()) == 2:
            s = ClassificationExperiment()
            s.setup(data, target = target1, session_id = 123)
    
            with st.spinner("Wait ..."):
                best = s.compare_models()
                st.write("The best classification model that you can use for prediction is:",best.__str__())
        else:
            s = RegressionExperiment()
            s.setup(data, target = target1, session_id = 123)
    
            with st.spinner("Wait ..."):
                best = s.compare_models()
                st.write("The best Regression model that you can use for prediction is:",best.__str__())
        model = best
  # select use your model and use paycart
        st.header("Test your model ")  
        st.write(s.predict_model(model))
        st.header("Save your Model")
        def download_model(model):
            output_model = pickle.dumps(model)
            b64 = base64.b64encode(output_model).decode()
            href = f'<a href="data:file/output_model;base64,{b64}" download="myfile.pkl">Download Trained Model .pkl File</a>'
            st.markdown(href, unsafe_allow_html=True)
        download_model(model)



