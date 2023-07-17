import pandas as pd
import numpy as np
import math
from datetime import datetime
import pandas_datareader as data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st

start='2010-01-01'
end= '2019-12-31'

st.title('STOCK PRICE PREDICTION SYSTEM')

user_input = st.text_input('Enter Stock', 'TATAMOTORS.NS')
df = yf.download(user_input, start, end,)

st.subheader('Data from 2010-2019')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (16,8))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (16,8))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (16,8))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
scaler = MinMaxScaler(feature_range= (0,1))

data_training_array= scaler.fit_transform(data_training)

x_train=[]
y_train=[]

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])
#

x_train = np.array(x_train)
y_train = np.array(y_train)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR', fontsize=18)
plt.plot(y_test, 'r', label = 'Original price')
plt.plot(y_predicted, 'g', label = 'Predicted price')
plt.legend(['Original Price', 'Predicted Prices'])
st.pyplot(fig2)