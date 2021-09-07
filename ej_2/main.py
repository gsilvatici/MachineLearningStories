import math
import numpy as np
import operator
import pandas as pd


class Category:

    def __init__(self, name):
        self.name = name
        self.headline_count = 0
        self.probability = 0
        self.words = {}
        self.relative_frequencies = {}
        self.probability_of_no_data = 0

    def add_words_from_headline(self, headline):
        
        self.headline_count += 1

        keys = headline.lower().split()
        key_set = list(set(keys))

        for key in key_set:
        	# processed[key] = keys.count(key)
        	if key not in self.words.keys():
	            self.words[key] = 0
	        self.words[key] = self.words.get(key) + keys.count(key)  


    def learn(self, total_headlines):
	    # probability of each word
	    k = len(self.words.keys())
	    total = sum(self.words.values())
	    self.probability = self.headline_count / total_headlines
	    self.probability_of_no_data = 1 / (total + k)

	    for word, cardinal in self.words.items():
	    	self.relative_frequencies[word] = (cardinal + 1) / (total + k)


    def get_productorial(self, headline):
        words = headline.lower().split()

        prod = 1

        for word in words:
            prod = prod * self.probability_of_no_data if word not in self.relative_frequencies.keys() \
                else self.relative_frequencies.get(word)

        # for word in words:
        # 	if word not in self.relative_frequencies.keys():
        # 		prod = prod * self.probability_of_no_data
        # 	else:
        # 		prod = prod * self.relative_frequencies.get(word)

        return prod * self.probability


class Bayes:

    def __init__(self, df):
        self.categories = {}
        self.confusion_matrix = {}

        for headline, category in zip(df['titular'], df['categoria']):
	        if category not in self.categories.keys():
	            self.categories[category] = Category(category)
	        self.categories[category].add_words_from_headline(headline)

    def learn(self, total_headlines):
    	for category in self.categories.values():
    		category.learn(total_headlines)
    	print("Bayes learning finished.")


    def classify(self, testing_data):

        self.confusion_matrix = {
            category: {category: 0 for category in self.categories.keys()}
            for category in self.categories.keys()
        }

        for i in range(len(testing_data)):
            row = testing_data.iloc[i]
            headline = row.titular
            category = row.categoria

            productorial = {name: category.get_productorial(headline) for name, category in self.categories.items()}
            
            winner = max(productorial.items(), key = operator.itemgetter(1))[0]

            self.confusion_matrix[category][winner] += 1


def split_dataframe(df, percentage):

    msk = np.random.rand(len(df)) < percentage
    training_data = df[msk]
    testing_data = df[~msk]

    return training_data, testing_data


news_data = pd.read_excel('Noticias_argentinas.xlsx')

news_data = news_data.loc[news_data['categoria'].notnull()]

categories_filter = [
    'Salud',
    'Deportes',
    'Economia',
    'Ciencia y Tecnologia',
    'Entretenimiento',
    'Nacional',
    'Internacional'
]

news_data = news_data.loc[news_data['categoria'].isin(categories_filter)]

print(len(news_data.index))

training_data, testing_data = split_dataframe(news_data, percentage = 0.99)

bayes = Bayes(training_data)

bayes.learn(len(news_data.index))

bayes.classify(testing_data)

# print(bayes.confusion_matrix)
