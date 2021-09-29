import math

import numpy as np
from src.aux_functions import euclidean_distance_from

class KNN:

    def __init__(self, train_data, train_labels, classes, k=5):
        self.k = k
        self.train_data = train_data
        self.train_labels = train_labels
        self.classes = classes

    def classify(self, register):
        distances = list(map(euclidean_distance_from(register), self.train_data))

        distances_tuples = list(zip(distances, self.train_labels))

        distances_tuples.sort()
        knn = distances_tuples[:self.k]

        # Si el registro me dio distancia 0 es que es esta clase la que debe clasificar
        if knn[0][0] == 0.0:
            return knn[0][1]

        results = np.zeros(len(self.classes))
        for (distance, class_number) in knn: 
            results[class_number - 1] += self.weight(distance)

        max_value = np.max(results)
        winner = np.where(results == max_value)[0]


        return winner[0]+1  if len(winner) == 1 or self.k == len(self.train_data) else self.__step_k(register)

    def __step_k(self, register):
        knn = KNN(self.train_data, self.train_labels, self.classes, k=(self.k + 1))
        aux = knn.classify(register)
        # print('aux: ' + str(aux) )
        if aux is not None: 
            return aux

    @staticmethod
    def weight(distance):
        return 1

    def batch_classify(self, batch):
        return list(map(self.classify, batch))

class WeightedKNN(KNN):

    @staticmethod
    def weight(distance):
        return 1/(distance ** 2) if distance != 0 else math.inf
