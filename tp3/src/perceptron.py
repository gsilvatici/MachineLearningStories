import matplotlib.pyplot as plt
import numpy as np

from abc import ABC, abstractmethod


class Perceptron(ABC):
    def __init__(
        self,
        training_set,
        expected_output,
        learning_rate=0.001,
        limit=100,
        epsilon=1e-15,
    ):

        self.training_set = np.array(training_set)
        self.expected_output = np.array(expected_output)
        self.learning_rate = learning_rate
        self.limit = limit
        self.epsilon = epsilon
        self.current_error = float("inf")
        self._weights, self._min_weights, self._min_error = self._fit()

    @property
    def weights(self):
        return self._weights

    @property
    def min_weights(self):
        return self._min_weights

    @property
    def min_error(self):
        return self._min_error

    @abstractmethod
    def activation_function(self, h):
        pass

    @abstractmethod
    def error(self, x, y, w):
        pass

    def _fit(self):
        x = self.training_set
        y = self.expected_output
        epsilon = self.epsilon

        x = np.c_[x, np.ones(x.shape[0])]
        m, n = x.shape
        w = np.zeros(n)
        i = 0
        error = 1
        min_weights = None
        min_error = float("inf")
        errors = []

        while error > epsilon and i < self.limit:
            i_x = np.random.randint(0, m)
            x_i = x[i_x]
            y_i = y[i_x]
            linear_output = np.dot(x_i, w)
            y_predicted = self.activation_function(linear_output)
            delta_w = self.learning_rate * (y_i - y_predicted) * x_i
            w = w + delta_w
            error = self.error(x, y, w)
            self.current_error = error
            errors.append(error)

            if error < min_error:
                min_error = error
                min_weights = w

            i = i + 1

        return w, min_weights, min_error

    def predict(self, input_set):
        input_set = np.c_[input_set, np.ones(input_set.shape[0])]
        dot = np.dot(input_set, self.min_weights)
        return self.activation_function(dot)


class SimplePerceptron(Perceptron):
    def __init__(self, training_set, expected_output, **kwargs):
        super().__init__(training_set, expected_output, **kwargs)

    def activation_function(self, h):
        # return -1 if h < 0 else 1
        return np.where(h >= 0, 1, -1)

    def error(self, x, y, w):
        return np.sum(
            [
                abs(y[i] - self.activation_function(np.dot(x_i, w)))
                for i, x_i in enumerate(x)
            ]
        )
