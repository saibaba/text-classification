from collections import Counter
import math
import csv
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import RegexpTokenizer
import random
import operator
from collections import defaultdict


cachedStopWords = stopwords.words("english")

with open("corpus.csv", 'r') as file:
  reviews = list(csv.reader(file))

reviews = [ [r[0], [w for w in r[1].split() if w.isalpha()] ] for r in reviews ]

categories = sorted(set(r[0] for r in reviews))

random.shuffle(reviews)


n = len(reviews)
fraction = n * 3 / 4

test_reviews = reviews[fraction:]
reviews = reviews[1:fraction]

n = len(reviews)

print "size of reviews: ", n

print "sum of words in Pos docs (v1) : ", sum([ len(r[1]) for r in reviews if r[0] == 'Pos' ])

priors = dict((key, 0) for key in categories)

for r in reviews:
  priors[r[0]] = priors[r[0]]+1

for c in categories:
  priors[c] = priors[c]* 1.0 / n



# all possible words, vocabulary
vocab = set()

# features = { "<word>" : { "<category>" : { "count" : count, "prob" : 0  } } }
features = { }

def update_feature(word, category):
  if word in features:
    class_counts = features[word]
    if category in class_counts:
      features[word][category]["count"] = features[word][category]["count"] + 1
    else:
      features[word][category] =  { "count" : 1  }
  else:
    features[word] = { category  : { "count": 1 } }

for r in reviews:
  for w in r[1]:
    if w not in vocab:
      vocab.add(w)
    update_feature(w, r[0])

def count(w, category):
  if w in features and category in features[w]:
    return features[w][category]["count"]
  else:
    return 0


"""
      -----------------------------
      |   | Empty  |        |
      |   |        |  27    | ...         <- 'Pos' docs    P(murdered|Pos) = 27 / n1
      ----------------------- 
      |   |        |        |
      |   | ...    |  28    | ...         <- 'Neg' docs    P(murdered|Neg) = 28 / n2
      -----------------------------
           <w>      murdered              <- features

"""

def calculate_class_probabilities(category):
  # with Laplace smoothing
  log_all_wc = math.log( sum ( [ count(w, category) for w in vocab ] )  + len(vocab) )

  print " all words log for cat " , category, log_all_wc

  for w in vocab:
    p =  math.log(count(w, category) + 1.0 )  - log_all_wc
    if w not in features:
      features[w] = { category : { "count" : 0, "prob" : p } }
    else:
      if category not in features[w]:
        features[w][category] = { "count": 0, "prob" : p }
      else:
        features[w][category]["prob"] = p

print 'sum of all words in Pos docs = ' , sum ( [ count(w, 'Pos') for w in vocab ] )
print 'sum of murdered Pos docs = ' , count('murdered', 'Pos')
print 'vocab size =  ' , len(vocab)

# eg: prob(murdered|Pos) = #murdered + 1/#all_words_in_Pos+len(vocab)

for category in categories:
  calculate_class_probabilities(category)

def preprocess(sentence):
  sentence = sentence.lower()
  tokenizer = RegexpTokenizer(r'\w+')
  tokens = tokenizer.tokenize(sentence)
  filtered_words = [w for w in tokens if not w in cachedStopWords]
  alpha_words = [word for word in filtered_words if word.isalpha()] 
  return alpha_words

def calculate_posterior(words, category):
 
  text_counts = Counter(words) 
  p = 0.0
  for word in text_counts:
    if word in vocab:
      p += text_counts.get(word) * features[word][category]["prob"]
  return math.log(priors[category]) + p

def calculate_posteriors(text):
  posteriors = {}
  for category in priors.keys():
    posteriors[category] = calculate_posterior(text, category)
  return posteriors

def test_accuracy():

  confusion_matrix = defaultdict(lambda : defaultdict(int))
  n = 0
  for sample in test_reviews:
    n += 1
    posteriors = calculate_posteriors(sample[1])
    decided_category = max(posteriors.iteritems(), key=operator.itemgetter(1))[0]
    #print decided_category, "---->",  sample[0], "----->" , posteriors
    confusion_matrix[sample[0]][decided_category] += 1

  accuracy = 0.0
  miss_class_rate = 0.0

  print "------- Confusion Matrix -----" 
  for horiz_cat in sorted(confusion_matrix.keys()):
    print horiz_cat, "\t", 
    for vert_cat in sorted(confusion_matrix[horiz_cat].keys()):
      print confusion_matrix[horiz_cat][vert_cat], "\t",
      if (horiz_cat == vert_cat):
        accuracy += confusion_matrix[horiz_cat][vert_cat]
      else:
        miss_class_rate += confusion_matrix[horiz_cat][vert_cat]
    print ""

  accuracy /= n
  miss_class_rate /= n
  print "\t",
  for vert_cat in sorted(confusion_matrix.keys()):
    print vert_cat, "\t",
  print ""
     
  print "Accuracy:", accuracy
  print "Missclassification Rate:", miss_class_rate

  ci_factor = 1.96 * math.sqrt(accuracy*(1.0-accuracy)/n)  
  ci_low = accuracy - ci_factor
  ci_high = accuracy + ci_factor
  print "Confidence interval = [", ci_low, ",", ci_high, "]"
  print "------------------------------" 

test_accuracy()

text = """
lousy movie
"""

print "For prediction request: ", max(calculate_posteriors(preprocess(text)).iteritems(), key=operator.itemgetter(1))[0]
