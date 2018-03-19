import numpy as np
import os

import matplotlib.pyplot as plt
import math
import random
import pprint as pp
import quant as q
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from keras import backend as K
from keras.optimizers import SGD,Adam
from sklearn.decomposition import PCA
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Bidirectional, Lambda
from functools import reduce

# fix random seed for reproducibility
np.random.seed(90210)


    
 
subset = 10000

path =r'/home/suroot/Documents/train/reg/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
data, debug = q.loadData(path, subset, loadDebug=False)
print(data.shape)

# verify that a random data point looks correct.
#sample = random.randint(0,subset-1)
#print("sample " + str(sample))
#q.debugPlot(data[sample,:], debug[sample])

batch_size=128
nb_epoch = 5

scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
data = scaler.fit_transform(data)

y = data[:,0]
y = np.reshape(y, (y.shape[0],1))
X = data[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(X_train.shape))

left_input = Input(X.shape)
right_input = Input(X.shape)
#build dense to use in each siamese 'leg'
model = Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(128, activation="relu"))


#call the convnet Sequential model on each of the input tensors so params will be shared
encoded_l = model(left_input)
encoded_r = model(right_input)
#layer to merge two encoded inputs with the l1 distance between them
L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
#call this layer on list of two input tensors.
L1_distance = L1_layer([encoded_l, encoded_r])
prediction = Dense(1,activation='sigmoid',bias_initializer=b_init)(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.00006)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer)

siamese_net.count_params()

siamese_net.summary()

# Train
history = model.fit(X_train, y_train, epochs=nb_epoch, batch_size=batch_size, shuffle=True, verbose=1)

# Evaluate train
#evaluation = model.evaluate(X_train, y_train, batch_size=batch_size, verbose=2)
#print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

# Evaluate test
evaluation = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=2)
print('Summary: Loss over the test dataset: %.2f, Accuracy: %.2f' % (evaluation[0], evaluation[1]))

y_hat = model.predict(X_test, verbose=1)
print(y_test.shape)
print(y_hat.shape)
plt.scatter(y_test, y_hat) 
plt.show()  
residuals = np.c_[y_test, y_hat]
print(residuals)

json_string = model.to_json()
with open(path+"/model.json", 'w') as text_file:
    print(json_string, file=text_file)

model.save_weights(path+"/weights.hdf5")

