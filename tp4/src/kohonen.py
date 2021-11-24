import math
import numpy as np


class Kohonen(object):
    def __init__(self, input_size, k, learning_rate=0.1, r=None, epochs=None):
        self.k = k
        self.input_size = input_size
        self.output_neurons = np.random.uniform(-1, 1, size=(k, k, input_size))
        self.learning_rate = learning_rate
        self.init_learning_rate = learning_rate
        self.r = r or k
        self.init_r = self.r
        self.epochs = epochs or (500 * input_size)

    def fit(self, X):
        epoch = 0
        while epoch < self.epochs:
            for x  in X:
                row_idx, col_idx = self.__most_similar_neuron(x)
                self.output_neurons[(row_idx, col_idx)] = self.__update_winner_weight(
                    x, row_idx, col_idx
                )
                neighbors_coordinates = self.__get_neighbors(row_idx, col_idx)
                for row, col in neighbors_coordinates:
                    d = np.linalg.norm(
                        np.array([row, col]) - np.array([row_idx, col_idx])
                    )
                    self.output_neurons[(row, col)] = self.__update_neighbor_weight(
                        x, row, col, d
                    )
            epoch += 1
            self.__update_r(epoch)
            self.__update_learning_rate(epoch)
            if epoch % 5 == 0:
                print(f"Epoch: {epoch}")

    def predict(self, x):
        return self.__most_similar_neuron(x)

    def __most_similar_neuron(self, x):
        diff = self.output_neurons - x
        norms = np.linalg.norm(diff, axis=2)

        return np.unravel_index(np.argmin(norms, axis=None), norms.shape)

    def __get_neighbors(self, center_y, center_x):
        neighborhood = set(
            [
                (i, j)
                for i in range(self.k)
                for j in range(self.k)
                if math.sqrt((i - center_y) ** 2 + (j - center_x) ** 2) < self.r
            ]
        )

        # Remove the center as it's not a neighbor
        return neighborhood - set([(center_y, center_x)])

    def __update_winner_weight(self, input, row, col):
        weights = self.output_neurons[(row, col)]

        return weights + self.learning_rate * (input - weights)

    def __update_neighbor_weight(self, input, row, col, distance):
        weights = self.output_neurons[(row, col)]
        V = np.exp(-2 * distance / self.r)
        delta_w = V * self.learning_rate * (input - weights)
        return weights + delta_w

    def __update_r(self, epoch):
        return (self.epochs - epoch) * self.init_r / self.epochs

    def __update_learning_rate(self, epoch):
        return self.init_learning_rate * (1 - (epoch / self.epochs))

