from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
import yfinance
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from sklearn.metrics import precision_score, r2_score, mean_absolute_percentage_error, accuracy_score,mean_squared_error

ticker = "TSLA"
# df = yfinance.download(ticker, period="1y")
# df.to_csv("Train.csv")


df = pd.read_csv('Train.csv')
print(df.head())
df = df.head(150)

cols = list(df)[2:12]
print(cols)

df_for_training = df[cols].astype(float)

scaler = StandardScaler()
scaler = scaler.fit(df_for_training)
df_for_training_scaled = scaler.transform(df_for_training)

trainX = []
trainY = []

# Number of days we want to look into the future based on the past days.
n_future = 1
n_past = 14  # Number of past days we want to use to predict the future.

for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
    trainX.append(
        df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 9])

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

df_for_training_scaled

trainY

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(
    trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(trainY.shape[1]))

model.compile(optimizer='adam', loss='mse')
model.summary()

history = model.fit(trainX, trainY, epochs=5, batch_size=3,
                    validation_split=0.1, verbose=1)

plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()

us_bd = CustomBusinessDay(calendar=USFederalHolidayCalendar())

df2 = df.head(150)
train_dates = pd.to_datetime(df2['Date'])
print(train_dates.tail(15))

n_past = 1
n_days_for_prediction = 50  # let us predict past 50 days

predict_period_dates = pd.to_datetime(df['Date'].tail(50)).tolist()
# predict_period_dates = pd.date_range(
#     list(train_dates)[-n_past], periods=n_days_for_prediction, freq='M').tolist()
print(predict_period_dates)

prediction = model.predict(trainX[-n_days_for_prediction:])
prediction_copies = np.repeat(prediction, df_for_training.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(prediction_copies)[:, 9]

forecast_dates = []
for time_i in predict_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame(
    {'Date': np.array(forecast_dates), 'price': y_pred_future})
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])


original = df[['Date', 'price']].iloc[-50:]
original['Date'] = pd.to_datetime(original['Date'])
print(original)
print(df_forecast)

original.plot(x='Date', y='price')
# original = pd.concat([original, df_forecast], axis=0) kyu bhai ??
original.plot(x='Date', y='price')

error = mean_absolute_error(original['price'], df_forecast['price'])
mse = mean_squared_error(original['price'], df_forecast['price'])
# precision = precision_score(original['price'],df_forecast['price'])
r2 = r2_score(original['price'], df_forecast['price'])
mape = mean_absolute_percentage_error(original['price'], df_forecast['price'])
# acc = accuracy_score(original['price'], df_forecast['price'])
print("*******************************************")
print("R2 is ", r2)
print("Mean absolute error is : ", error)
print("Mean square error is : ", mse)
print("MAPE is ", mape)
# print("Accuracy is ", acc)    showing error 
print("*******************************************")
