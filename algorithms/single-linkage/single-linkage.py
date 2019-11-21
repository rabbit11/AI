import seaborn as sns
import pandas as pd
import math
import os
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score

print("Escolha o arquivo 1, 2 ou 3 de entrada")

val = int(input())

readClu = []
resultClu = []

# selecionando qual arquivo usar segundo entrada
if val == 1:
    c = pd.read_csv("../../datasets/datasets/c2ds1-2sp.csv", sep='\t')
    readClu = open("../../datasets/datasets/c2ds1-2spReal.clu", 'r')
    kmin = 2
    kmax = 5
elif val == 2:
    c = pd.read_csv("../../datasets/datasets/c2ds3-2g.csv", sep='\t')
    readClu = open("../../datasets/datasets/c2ds3-2gReal.clu", 'r')
    kmin = 2
    kmax = 5
else:
    c = pd.read_csv("../../datasets/datasets/monkey.csv", sep='\t')
    readClu = open("../../datasets/datasets/monkeyReal1.clu", 'r')
    kmin = 5
    kmax = 12

# alocando os valores dos arquivos em uma variavel auxiliar
X = c.iloc[:, [1, 2]].values

# retirando dos arquivos .clu, informacoes desnecessarias
for linha in readClu:
    novaLinha = linha.strip("\n").split("\t")
    resultClu.append(int(novaLinha[1]))

# algoritmo single-link de fato, que recebe como parametro o numero minimo e o maximo de iteracoes
for x in range(kmin, kmax + 1):

    al = AgglomerativeClustering(
        n_clusters=x, affinity='euclidean', linkage='single')

    data = al.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], s=100, c=al.labels_.astype(
        float), label='Cluster ' + str(x - kmin))

    if val == 1:
        title = 'c2ds1-2sp ' + str(x) + 'c'
        plt.title(title)
        # plt.savefig(title)
    elif val == 2:
        title = 'c2ds3-2g ' + str(x) + 'c'
        plt.title(title)
        # plt.savefig(title)
    else:
        title = 'monkey ' + str(x) + 'c'
        plt.title(title)
        # plt.savefig(title)

    # print(data)
    # print(resultClu)

    # obtendo o indice rand
    print(adjusted_rand_score(data, resultClu))

    plt.show()

    # transformando os dados para dataframe
    df = pd.DataFrame(X)
    df.columns = ["d1", "d2"]
    df["cluster"] = data

    # podemos ou nao ordenar os dados segundo algum criterio
    df = df.sort_values("cluster")

    # salvando em csv o resultado do algoritmo
    df.to_csv("Resul-Single-dados" + str(val) +
              " e " + str(x) + "-clusters.csv")
