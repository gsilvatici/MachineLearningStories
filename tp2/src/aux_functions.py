import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

results_dir = './results'

def confusion_matrix(predictions, f_X, classes):
    conf_matrix = np.zeros((len(classes), len(classes)))
    for prediction, truth in zip(predictions, f_X):
        conf_matrix[int(int(truth) - 1)][int(int(prediction) - 1)] += 1

    return conf_matrix

def plot_matrix(matrix, filename):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for row, col in np.ndindex(matrix.shape):
        discr = matrix[row][col]
        ax.text(row, col, str(discr), va='center', ha='center')

    # plt.xlabel("Estimacion")
    # plt.ylabel("Real")
    plt.savefig(f'{filename}', bbox_inches='tight')
    plt.close()

def plot_precision(precisions_knn, precisions_weight_knn, k, filename):

    plt.title(f'Precisión con KNN pesado y k={k}')
    plt.ylabel('Precisión')
    plt.xlabel('Iteración del bloque cruzado')

    points = np.arange(len(precisions_weight_knn))
    width = 0.8

    plt.bar(points, precisions_weight_knn, width, label='KNN pesado', color='blue')

    plt.xticks(points)
    # plt.legend(loc='upper left')

    # plt.show()
    plt.savefig(f'{filename}')
    plt.close()


def normalize_df(data):
    normalized = data.copy()
    for attribute in data.columns:
        max_value = data[attribute].max()
        min_value = data[attribute].min()
        normalized[attribute] = (data[attribute] - min_value) / (max_value - min_value)
    return normalized


def euclidean_distance_from(x1):
    def euclidean_distance(x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return math.sqrt(distance)
    return euclidean_distance

def plot_precision_k(knn_means, w_knn_means, labels, filename):

    points = np.arange(len(knn_means))

    # plt.title('Precision obtenida sobre distintos K de validacion cruzada' )
    # plt.ylabel('Precisión')
    # plt.xlabel('K de validacion cruzada')
    # plt.xticks(points, labels)


    # points = np.arange(len(knn_means))
    # width = 0.8

    # plt.bar(points, knn_means, 0.8, color='blue',label='KNN means')

    # plt.bar(points, weight_knn_means, 0.5, color='red', label='Weighted KNN means')

    # plt.errorbar(
    #     points, knn_means, knn_stds, linestyle='None', label='KNN', capsize=5, marker='o', color='blue'
    # )
    # plt.errorbar(
    #     points, w_knn_means, w_knn_stds, linestyle='None', label='Weighted KNN', capsize=5, marker='o', color='plum'
    # )
    # plt.legend()

    # plt.savefig(f'{filename}', bbox_inches='tight')
    # plt.clf()

    plt.title('Precision obtenida sobre distintos K de validacion cruzada' )
    plt.ylabel('Precisión')
    plt.xlabel('K de validacion cruzada')

    plt.plot(knn_means, label='KNN', color='red')
    plt.plot(w_knn_means,  label='KNN pesado', color='blue')
    plt.xticks(points, labels)
    plt.legend()

    # filename = filename.split('.')[0] + '_only_means.png'
    plt.savefig(f'{filename}')
    plt.close()
