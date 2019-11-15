import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.cluster import adjusted_rand_score
import csv
import scipy.cluster.hierarchy as shc

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')


def main():
    real_value = []
    ARI = []
    i = 0

    # Pergunta o arquivo desejado
    print(""" 
    1. c2ds1-2sp.txt
    2. c2ds3-2g.txt
    3. monkey.txt
    """)

    option = int(input("Enter the option: "))

    # Intervalo de valores para k (numero de clusters):
    #kMin = int(input("Kmin: "))
    #kMax = int(input("Kmax: "))

    if (option == 1):
        file_name = "../../datasets/datasets/c2ds1-2sp.txt"
        Realfile_name = "../../datasets/datasets/c2ds1-2spReal.clu"
        resultfile_name = "./results/single-link_results1.csv"
        k_min = 2
        k_max = 5
    elif (option == 2):
        file_name = "../../datasets/datasets/c2ds3-2g.txt"
        Realfile_name = "../../datasets/datasets/c2ds3-2gReal.clu"
        resultfile_name = "../../Resultados/single-link_results2.csv"
        k_min = 2
        k_max = 5
    elif (option == 3):
        file_name = "../../datasets/datasets/monkey.txt"
        Realfile_name = "../../datasets/datasets/monkeyReal1.clu"
        resultfile_name = "../../Resultados/single-link_results3.csv"
        k_min = 5
        k_max = 12

    # Abertura do arquivo como leitura
    read = open(file_name, 'r')

    # Leitura dos dados. Vale notar que os dados estao com '.' em vez de ',' portanto eh necessario modificar:
    X = []
    for line in read:
        aux = []
        newline = line.rstrip("\n").split("\t")
        newline[1].replace(".", ",")
        newline[2].replace(".", ",")
        if(i != 0):
            aux.append(float(newline[1]))
            aux.append(float(newline[2]))
            X.append(aux)
            #np.insert(X, aux)
        i = i + 1
    read.close()

    read = open(Realfile_name, 'r')

    # Separacao da segunda coluna do arquivo
    for line in read:
        newline = line.rstrip("\n").split("\t")
        real_value.append(int(newline[1]))

    X = np.array(X)

    for j in range(k_min, k_max + 1):
        resultimage_name = "resultado" + str(option) + "_" + str(j) + ".png"
        resultfile_name = "./results/single-link_results1" + \
            str(option) + ".csv"
        dendoimage_name = "dendogram" + str(option) + "_" + str(j) + ".png"

        cluster = AgglomerativeClustering(
            n_clusters=j, affinity='euclidean', linkage='single')
        cluster.fit_predict(X)

        plt.figure(1)
        plt.clf()
        plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow', s=1)
        plt.savefig(resultimage_name, bbox_inches='tight')

        plt.figure(2)
        plt.clf()
        shc.dendrogram(shc.linkage(X, method='single'))
        plt.savefig(dendoimage_name, bbox_inches='tight')

        ARI.append(adjusted_rand_score(real_value, cluster.labels_))

    # Exporta para CSV
    with open(resultfile_name, 'w') as csvfile:
        fieldnames = ['K', 'ARI']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        i = 0
        for i in range(len(ARI)):
            writer.writerow({'K': i + k_min, 'ARI': ARI[i]})


if __name__ == "__main__":
    main()
