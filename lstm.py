import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math
import csv
import random
import json
import pprint as pp
import quant as q
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from functools import reduce

# fix random seed for reproducibility
np.random.seed(90210)

def loadData(path, subset = -1, loadDebug = False):        
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    if(subset > 0):
        allFiles = allFiles[0:subset]
    data = []
    debug = []
    for file in allFiles:
        print(file)
        fileNum = file.split('/')[-1].replace("data_","").replace(".csv","")
        with open(file, 'r') as f:
            data.append([float(val) for sublist in list(csv.reader(f)) for val in sublist]) 
        if loadDebug:       
            with open(file.replace("data_"+fileNum+".csv","debug_"+fileNum+".csv")) as f:
                debug.append(json.load(f))       
    return (np.array(data), debug)
    
 
subset = 10000

path =r'/home/suroot/Documents/train/reg/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
data, debug = loadData(path, subset, loadDebug=False)
print(data.shape)

# verify that a random data point looks correct.
#sample = random.randint(0,subset-1)
#print("sample " + str(sample))
#q.debugPlot(data[sample,:], debug[sample])

batch_size=128
nb_epoch = 5

scaler = MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

y = data[:,0]
y = np.reshape(y, (y.shape[0],1))
X = data[:,1:]
X = np.reshape(X, (X.shape[0],450,2))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(X_train.shape))


# create and fit the LSTM network
model = Sequential()
model.add(Bidirectional(LSTM(90, dropout=0.2, activation="tanh"), input_shape=(450, 2)) )
#model.add(LSTM(90, input_shape=(450, 2), dropout=0.2, activation="tanh") )
model.add(Dense(1, activation="linear"))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_percentage_error'])

model.summary()

# Train
history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)

# Evaluate train
#evaluation = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=2)
#print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

# Evaluate test
evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

