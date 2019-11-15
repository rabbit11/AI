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
    kmin = 2
    kmax = 5
elif val == 2:
    c = pd.read_csv("../../datasets/datasets/c2ds3-2g.csv", sep='\t')
    kmin = 2
    kmax = 5
else:
    c = pd.read_csv("../../datasets/datasets/monkey.csv", sep='\t')
    kmin = 5
    kmax = 12

nclusters = kmax - kmin

X = c.iloc[:, [1, 2]].values
al = AgglomerativeClustering(
    n_clusters=nclusters, affinity='euclidean', linkage='single')

data = al.fit_predict(X)

for i in range(kmin, kmax + 1):
    plt.scatter(X[:, 0], X[:, 1], c=al.labels_, cmap='rainbow', s=100)

    plt.title('Clusters Single-Linkage')
    plt.savefig('single-linkage' + str(i-1))
