import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

#nltk.download('punkt')

with open("intents.json") as file:
	data = json.load(file)


words = []
lables = []
docs_x = []
docs_y = []

for intent in data['intents']:
	for pattern in intent['patterns']:
		wrds = nltk.word_tokenize(pattern) 
		words.extend(wrds)
		docs_x.append(pattern)

		if intent['tag'] not in lables:
			lables.append(intent['tag'])



# print("words : ",words)

words = [stemmer.stem(w.lower()) for w in words]
words = sorted(List(set(words)))
# print("words : ",words)
# print("lables : ",lables)
# print("docs_x : ",docs_x)