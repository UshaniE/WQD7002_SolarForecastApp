# Import basic modules
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
#import pickle

from datetime import datetime
from datetime import timedelta

# Import regression and error metrics modules
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Standard scaler for preprocessing
from sklearn.preprocessing import StandardScaler

# Load specific forecasting tools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAXResults


#st.title("FORECASTING SOLAR PHOTOVOLTAICS (PV) POWER GENERATION USING MACHINE LEARNING")
st.header("Forecasting Solar Photovoltaics (PV) Power Generation Using Machine Learning")
st.write("Forecasting Horizon : **7-day ahead**")
st.write("Model : **SARIMAX**")

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
    
#date = st.sidebar.selectbox(label = "Select a Date")
# Enter start date
startdate = st.date_input(label = "Enter the start date for forecast",value=datetime(2013,9,1),
 min_value=datetime(2013,8,22), max_value=datetime(2013,12,24))
#st.write(start)


# Functions
# Train test split
def train_test(data, test_size = 0.15, scale = False, cols_to_transform=None, include_test_scale=False):
    """
    
        Perform train-test split with respect to time series structure
        
        - df: dataframe with variables X_n to train on and the dependent output y which is the column 'PowerAvg' in this notebook
        - test_size: size of test set
        - scale: if True, then the columns in the -'cols_to_transform'- list will be scaled using StandardScaler
        - include_test_scale: If True, the StandardScaler fits the data on the training as well as the test set; if False, then
          the StandardScaler fits only on the training set.
        
    """
    df = data.copy()
    # get the index after which test set starts
    test_index = int(len(df)*(1-test_size))
    
    # StandardScaler fit on the entire dataset
    if scale and include_test_scale:
        scaler = StandardScaler()
        df[cols_to_transform] = scaler.fit_transform(df[cols_to_transform])
        
    X_train = df.drop('PowerAvg', axis = 1).iloc[:test_index]
    y_train = df.PowerAvg.iloc[:test_index]
    X_test = df.drop('PowerAvg', axis = 1).iloc[test_index:]
    y_test = df.PowerAvg.iloc[test_index:]
    
    # StandardScaler fit only on the training set
    if scale and not include_test_scale:
        scaler = StandardScaler()
        X_train[cols_to_transform] = scaler.fit_transform(X_train[cols_to_transform])
        X_test[cols_to_transform] = scaler.transform(X_test[cols_to_transform])
    
    return X_train, X_test, y_train, y_test


# Forecasting on test set. 
# plot the predictions
def forecast(startdate,enddate):
    """
    Forecast 7 days ahead power generation on test set
    """
    # taking the input as date
    #startdate = datetime.strptime(start, "%Y-%m-%d")
    #enddate = startdate + timedelta(days=7)
    #startdateplot =  startdate - timedelta(days=7)

    y_actualsel = y_test.loc[startdate:enddate]
    X_predsel = X_test.loc[startdate:enddate]
    y_predsel = lr.predict(X_predsel)
    #y_actualplot =  y_test.loc[startdateplot:enddate]

    fig,axes = plt.subplots(figsize = (15,7))
    axes.plot(y_actualsel.index, y_actualsel, label='Observed')
    axes.plot(y_actualsel.index, y_predsel, color='r', label='Forecast')
    
    # set labels, legends and show plot
    axes.set_xlabel('Date')
    axes.set_ylabel('Power Generation in kW')
    axes.set_xticks(np.arange(0, len(y_actualsel.index), 24))
    axes.tick_params(axis='x',labelrotation = 90)
    axes.set

    axes.legend()

    st.subheader('Observed and Forecasted Power Generation')

    st.write(fig)  

    st.subheader('Model Performance Matrices')

    RMSE = round(np.sqrt(mean_squared_error(y_actual, pred2)),2)
    R2 = round(r2_score(y_actual, pred2),2)
    MAE = round(mean_absolute_error(y_actual, pred2),2)
  
    col1,col2,col3 = st.columns(3)
    col1.metric('RMSE',RMSE)
    col2.metric('R2',R2)
    col3.metric('MAE',MAE)

# Load data
dfcyc = pd.read_csv('Data_App.csv', index_col = 'Date', parse_dates=True)    

# Creating the training and test datasets

cols_to_transform = ['MeanWS', 'MaxGustWS', 'Precipitation', 'GlobalSolarRadiation',
       'TiltSolarIrradiance', 'DiffuseSolarIrradiance', 'OpenAirTempAvg',
       'ModuleTempAvg',]  
X_train_lag, X_test_lag, y_train_lag, y_test_lag = train_test(dfcyc, 
                                                              test_size = 0.15, scale = True, 
                                                              cols_to_transform=cols_to_transform)

# Since SARIMA model uses the past lag y values, scaling the power values as well. 
#i.e. fit the scaler on y_train and transform it and also transform y_test using the same scaler if required later

scaler1 = StandardScaler()
y_train_lag = pd.DataFrame(scaler1.fit_transform(y_train_lag.values.reshape(-1,1)), index = y_train_lag.index, 
                           columns = ['PowerAvg'])


loaded = SARIMAXResults.load('sarimax.pkl')

# Forecasting
forecast(startdate,enddate)




