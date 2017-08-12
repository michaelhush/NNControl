import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # removes warning
import pickle 
import KNN_class

## testing data
xtestset = pickle.load(open('xtestset.p','rb'))
ytestset = pickle.load(open('ytestset.p','rb'))
yltestset = pickle.load(open('yltestset.p','rb'))

num_of_samples = 100
maxtime = 2
timestep = 0.1
num_steps = int(maxtime/timestep)

## load plotting data
NN_plot = pickle.load(open('NN_plot.p','rb'))
LSTM_plot = pickle.load(open('LSTM_plot.p','rb'))
knn_plot = pickle.load(open('knn_plot.p','rb'))
RNN_plot = pickle.load(open('RNN_plot.p','rb'))
## Plot a sample test results
cmap =plt.get_cmap('hsv')
T = np.linspace(0, maxtime, num_steps)

## NN test plot
plt.figure(1)
plt.plot(T, NN_plot, '-', color=cmap(0.5), label = 'NN prediction')
## knn test plot
plt.plot(T, knn_plot, '-', color = cmap(0.6), label= 'KNN prediction')
## LSTM test plot
plt.plot(T, LSTM_plot, '-', color = cmap(0.7), label = 'LSTM prediction')
## linear test plot
lin_plot = np.reshape(yltestset[12], [num_steps]) ## change picture by changing the index of testset -- need to change for all test files
plt.plot(T, lin_plot, '-', color = cmap(0.1), label = 'Linear prediction') 
## RNN test plot
plt.plot(T, RNN_plot, '-', color = cmap(0.2), label = 'RNN prediction')
## True values
plt.plot(T, np.reshape(ytestset[12], [num_steps]), '-', color = cmap(0.8), label='Real output')
plt.xlabel('Time')
plt.ylabel('angular position')
plt.title('Predictions')
plt.legend()
plt.show()