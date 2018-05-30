from collections import Counter
import csv
import re
from nltk.corpus import stopwords
import nltk

from nltk.tokenize import RegexpTokenizer

cachedStopWords = stopwords.words("english")

def preprocess(sentence):
  sentence = sentence.lower()
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(sentence)
  filtered_words = [w for w in tokens if not w in cachedStopWords]
  alpha_words = [word for word in filtered_words if word.isalpha()] 
  return alpha_words

with open("movie-pang02.csv", 'r') as file:
  reviews = list(csv.reader(file))

#corpus = [ [r[0], nltk.word_tokenize(r[1].lower())] for r in reviews]
corpus = [ [r[0], preprocess(r[1])] for r in reviews]

for c in corpus:
  print c[0] +  "," + " ".join(c[1])


categories = sorted(set(r[0] for r in reviews))


#print categories
