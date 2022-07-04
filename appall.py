# Import basic modules
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import pickle
#import plotly.graph_objects as go

from datetime import datetime
from datetime import timedelta

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Standard scaler for preprocessing
from sklearn.preprocessing import StandardScaler

st.set_page_config(layout="wide")

st.header("Forecasting Solar Photovoltaics (PV) Power Generation Using Machine Learning")

#st.subheader("Select a Machine Learning Model")

st.sidebar.image("Picture1.jpg", use_column_width=True)

with st.sidebar:
     st.subheader("ML Models")
     st.write("   - Multiple Linear Regression")
     st.write("   - Ridge Linear Regression")
     st.write("   - Elastic Net Linear Regression")
     st.write("   - Random Forest")
     st.write("   - Seasonal AutoRegressive Integrated Moving Average with eXogenous (SARIMAX)")
     st.subheader("Most Suitable Model")
     st.write("SARIMAX")
     st.subheader("Data Source")
     st.write('Buruthakanda Solar Park, Sri Lanka')
     st.write('Hourly data from 2011-07-21 to 2013-12-31')



# Function for Multiple Linear Regression
def show_predict_MLR():  
# Load data
    X_test = pd.read_csv('X_test.csv', index_col = 'Date', infer_datetime_format=True)
    y_test = pd.read_csv('y_test.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
    with open('MLR.pkl' , 'rb') as f:
        lr = pickle.load(f)

    st.subheader("Multiple Linear Regression")

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

    # plot the predictions for last 7 days of the test set
    y_actualsel = y_test.loc['2013-12-23':'2014-01-01']
    X_predsel = X_test.loc['2013-12-23':'2014-01-01']
    y_actualplot = y_test.loc['2013-12-15':'2014-01-01']
    y_predsel = lr.predict(X_predsel)

    fig,axes = plt.subplots(figsize = (15,7))
    axes.plot(y_actualplot.index, y_actualplot, label='Observed')
    axes.plot(y_actualsel.index, y_predsel, color='r',  label='Forecast:MLR')

    # set labels, legends and show plot
    axes.set_xlabel('Date')
    axes.set_ylabel('Power Generation in kW')
    axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
    axes.tick_params(axis='x',labelrotation = 90)
    axes.legend()  

    st.write(fig)  

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Model Performance Matrices"}</h1>', unsafe_allow_html=True)

    # Performance evaluation
    pred = lr.predict(X_test)

    RMSE = round(np.sqrt(mean_squared_error(y_test, pred)),2)
    R2 = round(r2_score(y_test, pred),2)
    MAE = round(mean_absolute_error(y_test, pred),2)
  
    col1,col2,col3 = st.columns(3)
    col1.metric('RMSE',RMSE)
    col2.metric('R2',R2)
    col3.metric('MAE',MAE)

 # Function for Ridge Linear Regression
def show_predict_RLR():
# Load data
    X_test = pd.read_csv('X_test.csv', index_col = 'Date', infer_datetime_format=True)
    y_test = pd.read_csv('y_test.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
    with open('Ridge.pkl' , 'rb') as f:
        rr = pickle.load(f)

    st.subheader("Ridge Linear Regression")

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

# plot the predictions for last 7 days of the test set
    y_actualsel = y_test.loc['2013-12-23':'2014-01-01']
    X_predsel = X_test.loc['2013-12-23':'2014-01-01']
    y_actualplot = y_test.loc['2013-12-15':'2014-01-01']
    y_predselrr = rr.predict(X_predsel)

    fig,axes = plt.subplots(figsize = (15,7))
    axes.plot(y_actualplot.index, y_actualplot, label='Observed')
    axes.plot(y_actualsel.index, y_predselrr, color='y',  label='Forecast:RR')

# set labels, legends and show plot
    axes.set_xlabel('Date')
    axes.set_ylabel('Power Generation in kW')
    axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
    axes.tick_params(axis='x',labelrotation = 90)
    axes.legend()  

    st.write(fig)  

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Model Performance Matrices"}</h1>', unsafe_allow_html=True)

# Performance evaluation
    pred = rr.predict(X_test)

    RMSE = round(np.sqrt(mean_squared_error(y_test, pred)),2)
    R2 = round(r2_score(y_test, pred),2)
    MAE = round(mean_absolute_error(y_test, pred),2)
  
    col1,col2,col3 = st.columns(3)
    col1.metric('RMSE',RMSE)
    col2.metric('R2',R2)
    col3.metric('MAE',MAE)   

# Functions for Elastic Net Regression
def show_predict_ENR():
 
# Load data
    X_test_RF = pd.read_csv('X_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)
    y_test_RF = pd.read_csv('y_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
    with open('ElasticNetl.pkl' , 'rb') as f:
        enr= pickle.load(f)

    st.subheader("Elastic Net Regression")
    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

# plot the predictions for last 7 days of the test set
    X_predselrfl = X_test_RF.loc['2013-12-23':'2014-01-01']
    y_predselenr = enr.predict(X_predselrfl)
    y_actualsel = y_test_RF.loc['2013-12-23':'2014-01-01']
    y_actualplot = y_test_RF.loc['2013-12-15':'2014-01-01']
       

    fig,axes = plt.subplots(figsize = (15,7))
    axes.plot(y_actualplot.index, y_actualplot, label='Observed')
    axes.plot(y_actualsel.index, y_predselenr, color='c',  label='Forecast:ENR')

# set labels, legends and show plot
    axes.set_xlabel('Date')
    axes.set_ylabel('Power Generation in kW')
    axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
    axes.tick_params(axis='x',labelrotation = 90)
    axes.legend()  

    st.write(fig)  

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Model Performance Matrices"}</h1>', unsafe_allow_html=True)

# Performance evaluation
    predenr = enr.predict(X_test_RF)

    RMSE_ENR = round(np.sqrt(mean_squared_error(y_test_RF, predenr)),2)
    R2_ENR = round(r2_score(y_test_RF, predenr),2)
    MAE_ENR = round(mean_absolute_error(y_test_RF, predenr),2)
  
    col1,col2,col3 = st.columns(3)
    col1.metric('RMSE',RMSE_ENR)
    col2.metric('R2',R2_ENR)
    col3.metric('MAE',MAE_ENR)    


# Functions for Random Forest
def show_predict_RF():

# Load data
    X_test_RF = pd.read_csv('X_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)
    y_test_RF = pd.read_csv('y_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
    with open('rflag.pkl' , 'rb') as f:
        rfl= pickle.load(f)


    st.subheader("Random Forest")

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

# plot the predictions for last 7 days of the test set
    X_predselrfl = X_test_RF.loc['2013-12-23':'2014-01-01']
    y_predselrfl = rfl.predict(X_predselrfl)
    y_actualsel = y_test_RF.loc['2013-12-23':'2014-01-01']
    y_actualplot = y_test_RF.loc['2013-12-15':'2014-01-01']
       
    fig,axes = plt.subplots(figsize = (15,7))
    axes.plot(y_actualplot.index, y_actualplot, label='Observed')
    axes.plot(y_actualsel.index, y_predselrfl, color='g',  label='Forecast:RF')

# set labels, legends and show plot
    axes.set_xlabel('Date')
    axes.set_ylabel('Power Generation in kW')
    axes.set_xticks(np.arange(0, len(y_actualplot.index), 24))
    axes.tick_params(axis='x',labelrotation = 90)
    axes.legend()  

    st.write(fig)  

    st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Model Performance Matrices"}</h1>', unsafe_allow_html=True)

# Performance evaluation
    predrfl = rfl.predict(X_test_RF)

    RMSE_RF = round(np.sqrt(mean_squared_error(y_test_RF, predrfl)),2)
    R2_RF = round(r2_score(y_test_RF, predrfl),2)
    MAE_RF = round(mean_absolute_error(y_test_RF, predrfl),2)
  
    col1,col2,col3 = st.columns(3)
    col1.metric('RMSE',RMSE_RF)
    col2.metric('R2',R2_RF)
    col3.metric('MAE',MAE_RF)




page = st.selectbox("Select a Machine Learning Model",("Multiple Linear Regression","Ridge Linear Regression","Elastic Net Regression","Random Forest"))
if page == 'Multiple Linear Regression':
    show_predict_MLR()
elif page == 'Ridge Linear Regression':
    show_predict_RLR()
elif page == 'Elastic Net Regression':
    show_predict_ENR()
else:
    show_predict_RF()