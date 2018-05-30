Naive Bayes
===========

```
D = word=w1, word=w2, ... list of words in the document
C = classifications c1, c2, ... (denoted by ck below)

Problem: Given D=given, want to compute P(C=ck|D=given) for each ck and pick the one with largest value (MAP)
as this minimizes classification error (MSE).

How do we do that? Bayes theorem says,
 P(C|D) = P(D|C) * P(C) / P(D) = P(D, C) / P(D)

We know RHS. Why?

P(D) = P(D|C=c1) * P(C=c1) + P(D|C=c2) * P(C=c2) ...
Since, we only need to compare each p(C=ck|D=given), do not need to worry about denominator, P(D=given) anyways.

We know priors P(C=ck) for each classification, ck. Why?
  P(C=ck) = #of words in class ck/total # of words 
  These priors are MLE of classes that can be used as priors as long as there are >100 random samples and 
  hence correct proportions for each class (See discussion on page 20 of Bishop).
  100 are good based on CI as per Tom M. Mitchell in "Machine Learning"

How about p(D=given|C=ck) for each ck?... here comes labeled training set to the rescue!

We look at training set and figure out how the features (D) are distributed for each C=ck.

The D=given in our case is D=<list of words in the documnet>.
For each case ck, what is the distribution of these same words in it?
P(word=w1|C=ck) = count(w1 in all documents labeled ck)/count(all words in all documents labeled ck) 
...

(interestingly, what does this represent?
  count(wi in C=ck) / count(wi across all ck)
)  

From chain rule for conditional probability

P(word=w1,word=w2, ... word=wm|C=ck) * P(C=ck)
        = P(word=w1,word=w2, ... word=wm,C=ck)
        = P(word=w1,word=w2, ... word=wm,C=ck)
        = count(word=w1, word=w2... C=ck)/count(word=w2, .... C=ck) * count(word=w2, ... C=ck)/total-n
        = P(word=w1|word=w2, .. C=ck) * P(word=w2, ... C=ck)
        = ...
        = P(word=w1|word=w2, ... C=ck) * P(word=w2|word=w3... C=ck) * ... * P(word=wn|C=ck) * P(C=ck)

This is exponential (think through the "cells" as in the first chapter of Chris Bishop) !!!
So,
Assume order of words does not matter (Bag of words assumption)
Assume occurrence of one word does not depend on occurrence of another word but only on class, so:
  P(word=w1|word=w2...C=ck) = P(word=w1|C=ck) and so on. (conditional independence assumption)

This is called naive assumption - and helps to remove the exponential complexity

These probabilities can be calculated from sample/training set. How?

 a) In the training set, count each word wi across all documents in label C=ck separately for each label.
 b) In the training set, count of all words for each C=ck label seprately
 c) Their ratio is P(wi|C=ck)
 d) Do this for all words
 More details: Say count(wi, ck) = count of word wi across all document labeled with ck
                    count(w, ck)  = count of all words across all documents labeled ck
               Then P(wi|C=ck) = count(wi, ck) / count(w, ck)

From training, calc:
 p(D|C) = P(C|D) * P(D) / P(C)

Some notes:
* If multiple occurances of a word is not clipped a single occurance, then we are using multinomial distribution
* Features/attributes = Vocabulary of size k; Each word corresponds to one of x1, ... xk in the multinomial distribution with value for each being its frequency. And x1+...+xk = n (doc size, obviously not same n across all samples)

For prediction (multinomial):
    P(C=ck|givenText) = P(C=ck) * product-for-each-vocab-entry-index-by-i [ P(wi|C=ck)^freqi ]
    Here freqi is the frequency of i-th vocabulary word 

Python version also demonstrates: smoothing (Laplace), underflow handling (use log), checking model error/correctness (confusion matrix)

Todo: precision and recall for each category, F1 score averaging these.
```
References
==========

* Also solve: http://www.cs.cmu.edu/~aarti/Class/10601/homeworks/hw2Solutions.pdf
* https://stackoverflow.com/questions/19560498/faster-way-to-remove-stop-words-in-python
* https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
* https://www.inf.ed.ac.uk/teaching/courses/inf2b/learnnotes/inf2b-learn-note07-2up.pdf
* https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
* http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/
* http://ataspinar.com/2015/11/16/text-classification-and-sentiment-analysis/
* https://stats.stackexchange.com/questions/51296/how-do-you-calculate-precision-and-recall-for-multiclass-classification-using-co
* http://www.cs.oberlin.edu/~aeck/Fall2017/CSCI374/Handouts/CSCI374_ConfidenceIntervals.pdf
