# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:43:50 2020

@author: avales
https://python-binance.readthedocs.io/en/latest/

"""
from binance.client import Client

api_key =""
api_secret = ""

client = Client(api_key, api_secret)



#%% GET and SAVE HISTORY
from datetime import datetime
import pandas as pd
import mplfinance as mpf #https://github.com/matplotlib/mplfinance

klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 week ago UTC")

"""
1499040000000,      # Open time
"0.01634790",       # Open
"0.80000000",       # High
"0.01575800",       # Low
"0.01577100",       # Close
"148976.11427815",  # Volume
1499644799999,      # Close time
"2434.19055334",    # Quote asset volume
308,                # Number of trades
"1756.87402397",    # Taker buy base asset volume
"28.46694368",      # Taker buy quote asset volume
"17928899.62484339" # Can be ignored
"""

clearedOHLC = []
time = []
for OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteAssetVolume, NumberOfTrades, TakerBuyAssetVolume, TakerBuyQuoteAssetVolume, Ignored in klines :
    clearedOHLC.append([float(Open), float(High), float(Low), float(Close), float(Volume)])
    time.append(datetime.fromtimestamp(int(OpenTime)/1000))
del OpenTime, Open, High, Low, Close, Volume, CloseTime, QuoteAssetVolume, NumberOfTrades, TakerBuyAssetVolume, TakerBuyQuoteAssetVolume, Ignored

dfOHLC = pd.DataFrame(clearedOHLC, columns = ['Open','High', 'Low', 'Close','Volume'], index=time)
mpf.plot(dfOHLC[-50:],type='candle',mav=(3,6,9),volume=True)

#dfOHLC.to_csv('ten_month_ago_1mn_step.csv',header=True)
dfOHLC.to_csv('one_week_ago_1mn_step.csv',header=True)


#%% LOAD CSV
dataset = pd.read_csv('ten_month_ago_1mn_step.csv', index_col=[0])
dataset.index = pd.to_datetime(dataset.index)

mpf.plot(dataset[-50:],type='candle',mav=(3,6,9),volume=True)

dataset.head()

#LSTM initiation
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

tf.random.set_seed(13)
dataset.plot(subplots=True)

#%%PREPROCESSING
TRAIN_SPLIT = 300000
dataset = dataset.values

dts_train_mean = dataset[:TRAIN_SPLIT].mean()
dts_train_std = dataset[:TRAIN_SPLIT].std()
dataset = (dataset-dts_train_mean)/dts_train_std


past_history = 60
future_target = 1
STEP = 1

#target = Open
#                                multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False)
x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0, TRAIN_SPLIT, past_history, future_target, STEP, single_step=True)

x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

print ('Single window of past history : {}'.format(x_train_single[0].shape))

BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 200
EPOCHS = 10

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

#%% MODEL
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)
# Save the entire model as a SavedModel.
!mkdir saved_model
single_step_model.save('saved_model/')

def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

plot_train_history(single_step_history,
                   'Single Step Training and validation loss')


#%% EVALUATION
def create_time_steps(length):
  return list(range(-length, 0))


def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

for x, y in val_data_single.take(3):
  plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(),
                    single_step_model.predict(x)[0]], 12,
                   'Single Step Prediction')
  plot.show()

#%% PREDICTION
coordinates = np.random.randint(0, 100, size=(1, 60, 5))

converted_to_tensor = tf.convert_to_tensor(coordinates, dtype=tf.float32)

prediction = single_step_model.predict(converted_to_tensor)[0]
