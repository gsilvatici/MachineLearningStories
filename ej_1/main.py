import pandas as pd
from naive_bayes import NaiveBayes

training_set = pd.read_excel("PreferenciasBritanicos.xlsx")

testing_set = pd.DataFrame(data=[[1,0,1,1,0], [0,1,1,0,1]],
                columns=["scones", "cerveza", "wiskey","avena","futbol"])


print("\n\nTraining data\n\n", training_set, "\n\n")
print("Testing data\n\n", testing_set, "\n\n")

naive_bayes = NaiveBayes()
naive_bayes.train(training_set, "Nacionalidad", laplace = True)
prediction = naive_bayes.classify(testing_set)

print("Prediction\n\n", prediction)