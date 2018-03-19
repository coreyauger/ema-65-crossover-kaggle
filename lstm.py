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
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
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
    
 
subset = 32

path =r'/home/suroot/Documents/train/reg/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
data, debug = loadData(path, subset, loadDebug=True)
print(data.shape)


# verify that a random data point looks correct.
sample = random.randint(0,subset-1)
print("sample " + str(sample))

q.debugPlot(data[sample,:], debug[sample])


# create and fit the LSTM network
'''
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
'''