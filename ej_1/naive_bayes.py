import pandas as pd

class NaiveBayes:

    def train(self, training_set, attribute, laplace = True):
        self.attribute = attribute
        attribute_column = training_set[attribute]
        class_cardinal = attribute_column.value_counts()
        self.class_frequency = class_cardinal / len(training_set)
        aggregate = training_set.groupby(attribute).sum()
        class_cardinal.index.name = attribute

        if laplace:    
            aggregate_laplace = aggregate + 1
            self.relative_freqs = aggregate_laplace.div(class_cardinal + 2, axis = 'index')
        else:
            self.relative_freqs = aggregate.div(class_cardinal, axis='index')

    def classify(self, testing_set):
        for column, row in testing_set.iterrows():
            conditional_probabilities = self.relative_freqs * row + (1 - self.relative_freqs) * (1 - row)
            productorial = conditional_probabilities.product(axis = 1)
            h = productorial * self.class_frequency

            testing_set.at[column, self.attribute] = h.idxmax()
            for vj, prob in h.items():
                    testing_set.at[column, "H(" + vj + ")"] = prob
             
        return testing_set

