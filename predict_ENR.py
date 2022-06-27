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

# Import regression and error metrics modules
#from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Standard scaler for preprocessing
from sklearn.preprocessing import StandardScaler



# Functions
def show_predict_ENR():
    ok = st.button("Elastic Net Regression")
    if ok:
        st.subheader("Elastic Net Regression")

        st.markdown(f'<h1 style="color:#696969;font-size:22px;">{"Observed and Forecasted Power Generation"}</h1>', unsafe_allow_html=True)

# Load data
        X_test_RF = pd.read_csv('X_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)
        y_test_RF = pd.read_csv('y_test_lag_RF.csv', index_col = 'Date', infer_datetime_format=True)

# load saved model
        with open('ElasticNetl.pkl' , 'rb') as f:
            enr= pickle.load(f)

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