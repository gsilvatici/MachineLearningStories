import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

results_dir = './results'


def confusion_matrix(predictions, f_X, classes):
    """
    2x2 matrix
    Columns: predicted positive and predicted negative
    Rows: actual positive and negative

    |----------| positive | negative |
    | positive |    TP    |    FN    |
    | negative |    FP    |    TN    |

    """
    conf_matrix = np.zeros((len(classes), len(classes)))
    for prediction, truth in zip(predictions, f_X):
        conf_matrix[int(truth) - 1][int(prediction) - 1] += 1

    return conf_matrix

def plot_matrix(matrix, filename):
    fig, ax = plt.subplots()
    ax.matshow(matrix, cmap=plt.cm.Blues)

    for row, col in np.ndindex(matrix.shape):
        discr = matrix[row][col]
        ax.text(row, col, str(discr), va='center', ha='center')

    plt.xlabel("Estimacion")
    plt.ylabel("Esperado")
    plt.savefig(f'{filename}', bbox_inches='tight')
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
