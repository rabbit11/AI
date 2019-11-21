import pandas as pd
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

print("Escolha o arquivo 1, 2 ou 3 de entrada")
val = int(input())
print("Escolha o numero de iterações do KMedias")
n_iter = int(input())

readClu = []
resultClu = []

if val == 1:
    c = pd.read_csv("../../datasets/datasets/c2ds1-2sp.csv", sep='\t')
    readClu = open("../../datasets/datasets/c2ds1-2spReal.clu", 'r')
    kmin = 2;
    kmax = 5;
    name = 'c2ds1-2sp'
elif val == 2:
    c = pd.read_csv("../../datasets/datasets/c2ds3-2g.csv", sep='\t')
    readClu = open("../../datasets/datasets/c2ds3-2gReal.clu", 'r')
    kmin = 2;
    kmax = 5;
    name = 'c2ds3-2g'
else:
    c = pd.read_csv("../../datasets/datasets/monkey.csv", sep='\t')
    readClu = open("../../datasets/datasets/monkeyReal1.clu", 'r')
    kmin = 5;
    kmax = 12;
    name = 'monkey'

for linha in readClu:
    novaLinha = linha.strip("\n").split("\t")
    resultClu.append(int(novaLinha[1]))

X = c.iloc[:, [1, 2]].values

for x in range(kmin, kmax + 1):

    al = KMeans(n_clusters = x, n_init = n_iter)
    data = al.fit_predict(X)
    centroids = al.cluster_centers_
    # print(data.astype(int))
    output_file = open("output.clu", 'w+')
    np.savetxt('output' + '_' + title + str(x) + '.clu', data.astype(int), fmt='%i', delimiter=",")

    plt.scatter(X[:, 0], X[:, 1], s=100,c=al.labels_.astype(float), label = 'Cluster ' + str(x - kmin))
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)

    if val == 1:
        title = 'c2ds1-2sp ' + str(x) + 'c'
        plt.title(title)
        plt.savefig(title)
    elif val == 2:
        title = 'c2ds3-2g ' + str(x) + 'c'
        plt.title(title)
        plt.savefig(title)
    else:
        title = 'monkey ' + str(x) + 'c'
        plt.title(title)
        plt.savefig(title)

    print(adjusted_rand_score(data, resultClu))

    plt.show()

    # transformando os dados para dataframe
    df = pd.DataFrame(X)
    df.columns = ["d1", "d2"]
    df["cluster"] = data

    # podemos ou nao ordenar os dados segundo algum criterio
    # df = df.sort_values("cluster")

    # salvando em csv o resultado do algoritmo
    df.to_csv("Resul-K-Means-dados" + str(val) + " e "+ str(x) + "-clusters.csv")
