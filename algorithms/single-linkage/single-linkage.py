import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from scipy.spatial import distance
import math
import os

print(os.listdir("../../datasets/datasets/"))

c1 = pd.read_csv("../../datasets/datasets/c2ds1-2sp.csv", names=['x0', 'x1'])

print(c1)

plt.scatter(c1['x0'],c1['x1'])

def singleDistance(clusters, clustQty):
    while len(clusters) is not clustQty:
        #inicializando a distancia entre os clusters como infinito
        minDist = clust1 = clust2 = math.inf
        #enumerando cada um dos clusters
        for clusterId, cluster in enumerate(clusters[:len(clusters)]):
            #enumerando cada um dos pontos dentro de um cluster
            for pontoId, ponto in enumerate(cluster):
                #selecionando os clusters que não sejam aquele que 
                # estamos analisando no momento
                for clusterId2, cluster2 in enumerate(clusters[(clusterId+1):]):
                    #selecionando todos os pontos deste cluster
                    for pontoId2, ponto2 in enumerate(cluster2):
                        #se a distancia entre estes pontos, for menor que a distancia minima
                        # ate o momento, sobrescrevemos esta distancia
                        distPontos = distance.euclidean(ponto, ponto2)
                        if distPontos < minDist:
                            # sobrescrevendo a distancia minima entre estes 2 pontos
                            minDist = distPontos
                            # definindo os clusters que serão unidos
                            clust1 = clusterId 
                            # este cluster sera destruido no fim da execucao
                            clust2 = clusterId2 + clusterId + 1
        # funcao extend une os dados em ambos os clusters
        clusters[clust1].extend(cluster[clust2])
        # agora retiramos o cluster2, ja que unimos ambos os clusters no clust1
        clusters.pop(clust2)
    # retorna os clusters formados apos a execucao do single distance
    return(clusters)

# clusterizacao hierarquica dos dados
def hierarchical(dados, clusterQty, metrica = 'single'):

    initClusters = []
    # inicializa cada ponto como um unico cluster
    for indice, linha in dados.iterrows():
        # inicializa a matriz com os dados
        initClusters.append([[linha['x0'], linha['x1']]])
    
    return singleDistance(initClusters, clusterQty)

clusters = hierarchical(c1, 4)
colors = ['blue', 'red', 'purple', 'teal']

for cluster_index, cluster in enumerate(clusters):
    for point_index, point in enumerate(cluster):
        plt.plot([point[0]], [point[1]], marker='o', markersize=3, color=colors[cluster_index])

X = c1.as_matrix()
# generate the linkage matrix
# using single link metric to evaluate 'distance' between clusters
# single_link = linkage(X, 'single')

# c, coph_dists = cophenet(single_link, pdist(X))
# c

