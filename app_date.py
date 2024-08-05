import streamlit as st
import pandas as pd
import numpy as np
import joblib

import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# models and scaler
lstm_model = joblib.load('lstm_model.keras')
gru_model = joblib.load('gru_model.keras')
scaler = joblib.load('scaler.pkl')

# dataset
df = pd.read_csv("/content/drive/MyDrive/AI Final Project/CFC_traded_sahres_2019_to_date.csv")
df['Daily Date'] = pd.to_datetime(df['Daily Date'], format='%d/%m/%Y')
df = df.sort_values('Daily Date')

# sequences function
def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length + 1):
        x = data[i:(i + seq_length)]
        xs.append(x)
    return np.array(xs)

# predictions function
def make_prediction(date, model):
    # find the index of the date in the dataframe
    date_index = df[df['Daily Date'] == date].index[0]
    
    # get the previous 30 days of data
    data = df.iloc[date_index-29:date_index+1]
    
    # scaling
    scaled_data = scaler.transform(data.drop(columns=['Daily Date']))
    
    # creating sequence
    X = create_sequences(scaled_data, 30)
    
    # prediction
    prediction = model.predict(X)
    
    # Inverse transform the prediction
    prediction = scaler.inverse_transform(np.concatenate((prediction, np.zeros((prediction.shape[0], scaled_data.shape[1] - 1))), axis=1))[:, 0]
    
    return prediction[0]

# Streamlit app
st.title('Stock Price Prediction App')

# Date input
date = st.date_input('Select a date')

    
    # check if date is in the dataset
if date not in df['Daily Date'].values:
    st.error('Selected date is not in the dataset. Please choose another date.')
else:
    # predictions
    lstm_pred = make_prediction(date, lstm_model)
    gru_pred = make_prediction(date, gru_model)
        
    # calculating average prediction
    avg_pred = (lstm_pred + gru_pred) / 2
        
    # getting the actual price at the time window
    actual_price = df[df['Daily Date'] == date]['Closing Price - VWAP (GH¢)'].values[0]
        
    # calculating the difference between the average predicted price and the actual price
    difference = avg_pred - actual_price
        
    # displaying the results
    st.write(f'Predicted Closing Price: GH¢ {avg_pred:.2f}')
    st.write(f'Actual Closing Price: GH¢ {actual_price:.2f}')
    st.write(f'Difference: GH¢ {difference:.2f}')
    
    # graph plotting
    start_date = date - timedelta(days=14)
    end_date = date + timedelta(days=14)
        
    plot_data = df[(df['Daily Date'] >= start_date) & (df['Daily Date'] <= end_date)]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(plot_data['Daily Date'], plot_data['Closing Price - VWAP (GH¢)'], label='Actual Price')
    ax.axhline(y=avg_pred, color='r', linestyle='--', label='Predicted Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (GH¢)')
    ax.set_title('Actual vs Predicted Closing Price')
    ax.legend()
        
    st.pyplot(fig)
