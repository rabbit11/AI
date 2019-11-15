import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

print("Escolha o arquivo 1, 2 ou 3 de entrada")
val = int(input())

if val == 1:
    c = pd.read_csv("../../datasets/datasets/c2ds1-2sp.csv", sep='\t')
    kmin = 2;
    kmax = 5;
elif val == 2:
    c = pd.read_csv("../../datasets/datasets/c2ds3-2g.csv", sep='\t')
    kmin = 2;
    kmax = 5;
else:
    c = pd.read_csv("../../datasets/datasets/monkey.csv", sep='\t')
    kmin = 5;
    kmax = 12;

nclusters = kmax - kmin

X = c.iloc[:, [1, 2]].values
al = AgglomerativeClustering(n_clusters = nclusters, affinity = 'euclidean', linkage ='average')

data = al.fit_predict(X)

if val == 1 or val == 2:
    plt.scatter(X[data==0, 0], X[data==0, 1], s=100, c='red', label ='Cluster 1')
    plt.scatter(X[data==1, 0], X[data==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[data==2, 0], X[data==2, 1], s=100, c='green', label ='Cluster 3')
else:
    plt.scatter(X[data==0, 0], X[data==0, 1], s=100, c='red', label ='Cluster 1')
    plt.scatter(X[data==1, 0], X[data==1, 1], s=100, c='blue', label ='Cluster 2')
    plt.scatter(X[data==2, 0], X[data==2, 1], s=100, c='green', label ='Cluster 3')
    plt.scatter(X[data==3, 0], X[data==3, 1], s=100, c='cyan', label ='Cluster 4')
    plt.scatter(X[data==4, 0], X[data==4, 1], s=100, c='black', label ='Cluster 5')
    plt.scatter(X[data==5, 0], X[data==5, 1], s=100, c='yellow', label ='Cluster 6')
    plt.scatter(X[data==6, 0], X[data==6, 1], s=100, c='magenta', label ='Cluster 7')

plt.title('Clusters Average Link')
plt.show()
