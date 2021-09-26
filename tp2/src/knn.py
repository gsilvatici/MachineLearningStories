from collections import Counter

import numpy as np


def distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KNN:
    def __init__(self, labeled_set, k=3):
        """
        labeled_set is a list of tuples (point, class)
        """
        self.k = k
        self.labeled_set = labeled_set

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = [(distance(x, x_i), v) for x_i, v in self.labeled_set]
        k_neighbors = sorted(distances)[: self.k]
        most_common = Counter([value for _, value in k_neighbors]).most_common(1)
        return most_common[0][0]
