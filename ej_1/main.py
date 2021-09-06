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
                    testing_set.at[column, "(" + vj + ")"] = prob
             
        return testing_set



training_set = pd.read_excel("PreferenciasBritanicos.xlsx")

testing_set = pd.DataFrame(data=[[1,0,1,1,0], [0,1,1,0,1]],
                columns=["scones", "cerveza", "wiskey","avena","futbol"])


print("\n\nTraining data\n\n", training_set, "\n\n")
print("Testing data\n\n", testing_set, "\n\n")

naive_bayes = NaiveBayes()
naive_bayes.train(training_set, "Nacionalidad", laplace = True)
prediction = naive_bayes.classify(testing_set)

print("Prediction\n\n", prediction)