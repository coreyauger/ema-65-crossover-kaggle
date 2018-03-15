
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import math
import csv
import random
import quant as q
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from functools import reduce


# fix random seed for reproducibility
np.random.seed(90210)

def loadData(subset = -1):    
    path =r'/home/suroot/Documents/train/reg/ema-65-crossover' # path to data
    allFiles = glob.glob(os.path.join(path, "data_*.csv"))
    if(subset > 0):
        allFiles = allFiles[0:subset]
    data = []
    for file in allFiles:
        with open(file, 'r') as f:
            data.append([float(val) for sublist in list(csv.reader(f)) for val in sublist])        
    return np.array(data)
    

def priceChangeToPrice(data, initial = 100):
    return list(reduce(lambda x,y:  x + [ x[-1]+(x[-1]*y) ], data, list([initial]) ) ) 

subset = 45

data = loadData(subset)
print(data.shape)

# verify that a random data point looks correct.
sample = random.randint(0,subset-1)
print("sample " + str(sample))
sample90Min = ((data[sample,:])[1:181])
sample90Min = sample90Min[::2]  # only want price
graph = priceChangeToPrice(sample90Min)

ema15 = q.calculateEma(graph, 15, usePriceAsInitial = True) 
ema65 = q.calculateEma(graph, 65, usePriceAsInitial = True) 
#print(ema15)
plt.plot(graph)
plt.plot(ema15)
plt.plot(ema65)
plt.show()


# create and fit the LSTM network
'''
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
'''