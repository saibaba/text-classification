// http://ataspinar.com/2015/11/16/text-classification-and-sentiment-analysis/
// http://stackoverflow.com/questions/10059594/a-simple-explanation-of-naive-bayes-classification
// https://www.dataquest.io/blog/naive-bayes-tutorial/

import java.util.*;
import java.util.stream.*;

/**
 * D = word=w1, word=w2, ... list of words in the document
 * C = classifications c1, c2, ... (denoted by ck below)
 *
 * Problem: Given D=given, want to compute P(C=ck|D=given) for each ck and pick the one with largest value (MAP)
 * as this minimizes classification error (MSE).
 * 
 * How do we do that? Bayes theorem says,
 *  P(C|D) = P(D|C) * P(C) / P(D) = P(D, C) / P(D)
 *
 * We know RHS. Why?
 *
 * P(D) = P(D|C=c1) * P(C=c1) + P(D|C=c2) * P(C=c2) ...
 * Since, we only need to compare each p(C=ck|D=given), do not need to worry about denominator, P(D=given) anyways.
 *
 * We know priors P(C=ck) for each classification, ck. Why?
 *   P(C=ck) = #of words in class ck/total # of words 
 *   These priors are MLE of classes that can be used as priors as long as there are >100 random samples and 
 *   hence correct proportions for each class (See discussion on page 20 of Bishop).
 *   100 are good based on CI as per Tom M. Mitchell in "Machine Learning"
 *
 * How about p(D=given|C=ck) for each ck?... here comes labeled training set to the rescue!
 *
 * We look at training set and figure out how the features (D) are distributed for each C=ck.
 *
 * The D=given in our case is D=<list of words in the documnet>.
 * For each case ck, what is the distribution of these same words in it?
 * P(word=w1|C=ck) = count(w1 in all documents labeled ck)/count(all words in all documents labeled ck) 
 * ...
 *
 * (interestingly, what does this represent?
 *   count(wi in C=ck) / count(wi across all ck)
 * )  
 *
 * From chain rule for conditional probability
 *
 * P(word=w1,word=w2, ... word=wm|C=ck) * P(C=ck)
 *         = P(word=w1,word=w2, ... word=wm,C=ck)
 *         = P(word=w1,word=w2, ... word=wm,C=ck)
 *         = count(word=w1, word=w2... C=ck)/count(word=w2, .... C=ck) * count(word=w2, ... C=ck)/total-n
 *         = P(word=w1|word=w2, .. C=ck) * P(word=w2, ... C=ck)
 *         = ...
 *         = P(word=w1|word=w2, ... C=ck) * P(word=w2|word=w3... C=ck) * ... * P(word=wn|C=ck) * P(C=ck)
 *
 * This is exponential (think through the "cells" as in the first chapter of Chris Bishop) !!!
 * So,
 * Assume order of words does not matter (Bag of words assumption)
 * Assume occurrence of one word does not depend on occurrence of another word but only on class, so:
 *   P(word=w1|word=w2...C=ck) = P(word=w1|C=ck) and so on. (conditional independence assumption)
 *
 * This is called naive assumption - and helps to remove the exponential complexity
 * 
 * These probabilities can be calculated from sample/training set. How?
 *
 *  a) In the training set, count each word wi across all documents in label C=ck separately for each label.
 *  b) In the training set, count of all words for each C=ck label seprately
 *  c) Their ratio is P(wi|C=ck)
 *  d) Do this for all words
 *  More details: Say count(wi, ck) = count of word wi across all document labeled with ck
 *                     count(w, ck)  = count of all words across all documents labeled ck
 *                Then p(wi|C=ck) = count(wi, ck) / coiunt(w, ck)
 *
 * From training, calc:
 *  p(D|C) = P(C|D) * P(D) / P(C)
 *
 * TODO: smoothing (Laplace), underflow (use log), check model error/correctness (confusion matrix)
 */

class Feature {
  String name;
  Map<String, Double> classProbabilities = new HashMap<>();
  Map<String, Integer> categoryCounts = new HashMap<>();

  public void incr(int count, String category) {
    if (categoryCounts.containsKey(category)) {
      categoryCounts.put(category, categoryCounts.get(category) + count);
    } else {
      categoryCounts.put(category, count);
    }
  }

  public void normalize(String category, int categoryTotal) {
    classProbabilities.put(category, (double)categoryCounts.get(category)/categoryTotal);
  }

  public Double getProbability(String category) {
    return classProbabilities.get(category);
  }
}

public class Classify {

  // from training data
  static Map<String, Feature> featureMap = new HashMap<>();

  public static void updateFeature(String name, int count, String category) {
    if (featureMap.containsKey(name)) {
      featureMap.get(name).incr(count, category);
    } else {
      Feature f = new Feature();
      f.name = name;
      f.incr(count, category);
      featureMap.put(name, f);
    }
  }


  static String[] words = new String[] { "lame", "awesome", "this", "is", "blog-post" };
  static int[] pos = new int[] { 10, 70, 50, 100, 10 };
  static int[] neu = new int[] { 20, 20, 500, 600, 90 };
  static int[] neg = new int[] { 70, 10, 50, 100, 10 };

  static int words_in_pos_docs = IntStream.of(pos).sum();
  static int words_in_neu_docs = IntStream.of(neu).sum();
  static int words_in_neg_docs = IntStream.of(neg).sum();

  static String category[]  = new String[]   { "positive", "neutral", "negative" };
  static int categoryTotals[]  = new int[]   { words_in_pos_docs, words_in_neu_docs, words_in_neg_docs };

  static double[] classPriors = new double[3];

  public static void train() {
    for (int i = 0; i < words.length; i++) {
      updateFeature(words[i], pos[i], "positive");
      updateFeature(words[i], neu[i], "neutral");
      updateFeature(words[i], neg[i], "negative");
    }

    for (int i = 0 ; i < category.length; i++) {
      for (Feature f : featureMap.values()) {
        f.normalize(category[i], categoryTotals[i]);
      }
    }

    int total = 0;
    for (int i = 0; i < pos.length; i++) {
      total += pos[i];
      total += neu[i];
      total += neg[i];
    }

    int tpos = 0;
    for (int i = 0; i < pos.length; i++) {
      tpos += pos[i];
    }

    int tneu = 0;
    for (int i = 0; i < neu.length; i++) {
      tneu += neu[i];
    }

    int tneg = 0;
    for (int i = 0; i < neg.length; i++) {
      tneg += neg[i];
    }

    classPriors[0] = 0.33 ; //(double)tpos/total;
    classPriors[1] = 0.33 ; //(double)tneu/total;
    classPriors[2] = 0.34 ; //(double)tneg/total;

    System.out.println("Feature prob for this in +ve class=  " + featureMap.get("this").classProbabilities.get("positive"));
    System.out.println("Feature prob for blog-post in +ve class=  " + featureMap.get("blog-post").classProbabilities.get("positive"));
    System.out.println("Feature prob for is in +ve class=  " + featureMap.get("is").classProbabilities.get("positive"));
    System.out.println("Feature prob for awesome in +ve class=  " + featureMap.get("awesome").classProbabilities.get("positive"));
    System.out.println("");
    System.out.println("");
         
    System.out.println("Positive prior = " + classPriors[0]);
    System.out.println("Neutral prior = " + classPriors[1]);
    System.out.println("Negative prior = " + classPriors[2]);
    System.out.println("");
    System.out.println("");
  }

  public static void decide(String[] seen) {
    double ppos = classPriors[0];
    double pneu = classPriors[1];
    double pneg = classPriors[2];
  
    for (int i = 0; i < seen.length; i++) {
      Feature f = featureMap.get(seen[i]);
      ppos *= f.classProbabilities.get("positive");
      pneu *= f.classProbabilities.get("neutral");
      pneg *= f.classProbabilities.get("negative");
    }

    System.out.println("positive = " + ppos);
    System.out.println("neutral = " + pneu);
    System.out.println("negative = " + pneg);
  }

  public static void main(String[] args) {
    train();
    decide(new String[] { "this", "blog-post", "is", "awesome" });
    decide(new String[] { "this", "blog-post", "is", "lame" });
  }
}
