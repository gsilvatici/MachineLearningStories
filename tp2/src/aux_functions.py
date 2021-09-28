import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing

results_dir = './results'


def normalize_df(data):
    result = data.copy()
    for attribute in data.columns:
        max_value = data[attribute].max()
        min_value = data[attribute].min()
        result[attribute] = (data[attribute] - min_value) / (max_value - min_value)
    return result

def euclidean_distance_from(x1):
    def euclidean_distance(x2):
        distance = 0
        for i in range(len(x1)):
            distance += (x1[i] - x2[i]) ** 2
        return math.sqrt(distance)
    return euclidean_distance