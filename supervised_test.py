import base64
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.impute import SimpleImputer

from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment
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
    # Multi-selection widget to choose columns to drop
    columns_to_drop = st.multiselect('Select columns to drop:', df.columns)
    data = data.drop(columns = columns_to_drop)
    st.write(data.head())

    tab1, tab2  = st.tabs(["scatter plot","histogram"])

    col_num = data.select_dtypes(include = np.number).columns.to_list()
    # Identify categorical and numerical columns
    categorical_columns = data.select_dtypes(include=['object', 'category']).columns


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

    # if you want dropna data 
    drop_data = st.selectbox("Do you want to dropna data :",["Dropna"],index=None)
    if drop_data is not None:
        data = data.dropna()
        st.write("Number  of value null of the data :" ,data.isna().sum())
        st.write("The shape of the data :" ,data.shape)
    #number of column null st.write(data.isna().sum())
    st.write("Number  of value null of the data :" ,data.isna().sum())
    #calcul number null of categorical features
    if categorical_columns is not None:
        #st.write("Number null of categorical features :" ,data.isna().sum())
        #  what do you want to do with categorical 
        method_freq = st.selectbox("Select method impute for categorical :",["most_frequent","constant"],index=None)
        if method_freq is not None:
            objet_impute = SimpleImputer(strategy=method_freq,missing_values=np.nan)
            data[categorical_columns] = objet_impute.fit_transform(data[categorical_columns])
            st.write(data.isna().sum())

    #calcul number null of numeric features
    if col_num is not None:
        #st.write("Number null of numeric features :" ,data.isna().sum())
        #  what do you want to do with numerical features
        method_frq = st.selectbox("Select method impute for numeric:",["mean","median"],index=None)
        if method_frq is not None:
            num_impute = SimpleImputer(strategy=method_frq,missing_values=np.nan)
            data[col_num] = num_impute.fit_transform(data[col_num])
            st.write(data.isna().sum())


    target1 = st.selectbox("Select the target of your model:",data.columns,index=None)
    if (target1 is not None):
      #  data[target1] = data[target1].astype(float)
        if len(data[target1].value_counts()) <= 10:
            st.write("This model is Classification ")
            s = ClassificationExperiment()
            s.setup(data, target = target1, session_id = 123)
    
            with st.spinner("Wait ..."):
                best = s.compare_models()
                st.write("The best classification model that you can use for prediction is:",best.__str__())
        else:
            st.write("This model is regression ")
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
