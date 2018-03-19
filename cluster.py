import numpy as np
import os

import matplotlib.pyplot as plt
import math
import random
import pprint as pp
import quant as q
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from itertools import groupby
from functools import reduce
import pickle

# fix random seed for reproducibility
np.random.seed(90210)

subset = 10000

path =r'/home/suroot/Documents/train/reg/22222c82-59d1-4c56-a661-3e8afa594e9a' # path to data
data, debug = q.loadData(path, subset, loadDebug=False)
print(data.shape)

#scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
#data = scaler.fit_transform(data)

pca = PCA(n_components=900)
pca.fit(data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()

y = data[:,0]
y = np.reshape(y, (y.shape[0],1))
X = data[:,1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

print("Y shape: "+str(y_train.shape))
print("X shape: "+str(X_train.shape))

n_clusters = 32

#kmeans = KMeans(n_clusters=2, random_state=0).fit(X_train, y_train)
#print(kmeans.cluster_centers_)

#pca = PCA(n_components=2).fit(X_train)
#pca_2d = pca.transform(X_train)
#plt.figure('Reference Plot')
#plt.scatter(pca_2d[:, 0], pca_2d[:, 1])
kmeans = KMeans(n_clusters=n_clusters, random_state=111).fit(X_train, y_train)
#plt.figure('K-means with 3 clusters')
#plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
#plt.show()


perds = kmeans.fit_predict(X_test)
score = np.c_[perds, y_test]
cluster = score[score[:,0].argsort()]
with open("./cluster.npy","w") as text_file:
    print(cluster.tolist(), file=text_file)

for i in range(n_clusters):
    c = list(filter(lambda x: x[0] == i, cluster))
    neg = len(list(filter(lambda x: x[1] <= 0, c)))
    pos = len(c)-neg
    print("cluster["+str(i)+"] has "+ str(pos) +" / "+str(neg)+"\n")

#s = pickle.dumps(kmeans)
#print(s)


