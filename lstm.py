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

def loadData(subset = -1, loadDebug = False):    
    path =r'/home/suroot/Documents/train/reg/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
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
    

 

subset = 38

data, debug = loadData(subset, loadDebug=True)
print(data.shape)


# verify that a random data point looks correct.
sample = random.randint(0,subset-1)
print("sample " + str(sample))

q.debugPlot(data[sample,:], debug[sample])

'''
sample1Min = ((data[sample,:])[0:181])
sample5Min = ((data[sample,:])[181:361])
#print(sample90Min)
print(sample1Min.shape)
sample1Min = sample1Min[1::2]  # only want price
sample5Min = sample5Min[::2]
#print(sample90Min.shape)
trigger = debug[sample]["Trigger"]["parent"][0]
trainingExampleId = debug[sample]["TrainingExample"]["id"]
symbol = debug[sample]["TrainingExample"]["symbol"]["sym"]

triggerData = [val for sublist in trigger["event"]["data"] for val in sublist]
priceData = list(filter(lambda x: x["$type"] == "m.q.PriceTs", triggerData ))
ema15Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 15, triggerData ))
ema65Data = list(filter(lambda x: x["$type"] == "m.q.EmaTs" and x["data"]["timePeriod"] == 65, triggerData ))

# we need to rewind these values through time now.
rewindPrice1 = q.rewindPriceChangeToPrice(sample1Min[::-1], initial=priceData[0]["data"]["close"])
rewindPrice5 = q.rewindPriceChangeToPrice(sample5Min[::-1], initial=priceData[0]["data"]["close"])
print("15: "+str(ema15Data[0]["data"]["ema"]))
print("65: "+str(ema65Data[0]["data"]["ema"]))
rewindEma15 = q.rewindEma(rewindPrice1, 15, startEma = ema15Data[0]["data"]["ema"]) 
rewindEma65 = q.rewindEma(rewindPrice1, 65, startEma = ema65Data[0]["data"]["ema"]) 
print("rewindPrice1: " + str(rewindPrice1[-1]))
print(rewindEma65)
print("rewindEma15: " + str(rewindEma15[-1]))
print("rewindEma65: " + str(rewindEma65[-1]))

enterPrice = priceData[0]["data"]["close"]
print("symbol: "+symbol)
print("Training Example: " + trainingExampleId)
print("enter price: " + str(enterPrice))
print("enter time: " + priceData[0]["time"])
graph1 = q.priceChangeToPrice(sample1Min, initial=rewindPrice1[-1])
graph5 = q.priceChangeToPrice(sample5Min, initial=rewindPrice5[-1])
#print(graph)
#print(graph[-1])
ema15 = q.calculateEma(graph1, 15, startEma=graph1[0]) 
ema65 = q.calculateEma(graph1, 65, startEma=graph1[0]) 
plt.plot(([graph1[0]] * 72*5) + graph1)
plt.plot(([graph1[0]] * 72*5) + ema65)
plt.plot(([graph1[0]] * 72*5) + ema15)
graph5 = [[x]*5 for x in graph5]
graph5 = [val for sublist in graph5 for val in sublist]
plt.plot(graph5)
plt.show()
'''

# create and fit the LSTM network
'''
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
'''