import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import argparse


######


def process_data(path, words="word", tag="tag", sentence="sentence"):
    df = pd.read_csv(path,  sep='\t', names=[words, tag, '365', 'X', sentence])
    df = df.drop(['365', 'X'], axis=1)
    df[sentence] = df[sentence].astype(str)

    X = df.drop([tag], axis=1)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    y = df.tag.values

    classes = np.unique(y)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=0)

    return df, classes, new_classes, X_train, X_test, y_train, y_test


def test_classifier(X_train, y_train, X_test, y_test, classes, new_classes, name, random_state=42, *args, **kwargs):
  try:
    classifier = name(*args, **kwargs, random_state=random_state)
  except TypeError: # some classifiers don't have random_state
    classifier = name(*args, **kwargs)
  print(classifier) # print classifier parameters
  classifier.partial_fit(X_train, y_train, classes)
  print(classification_report(y_pred=classifier.predict(X_test), y_true=y_test, labels=new_classes))



def main():
  parser = argparse.ArgumentParser() 
  parser.add_argument('-f', '--file', type=argparse.FileType('r'), help="Path to dataset to be processed for baselines", required=True)
  parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default ='sentence')")
  parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default ='word')")
  parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default = 'tag')")
  parser.add_argument('-m', '--model', choices=['Perceptron', 'SGDClassifier', 'MultinomialNB', 'PassiveAggressiveClassifier'], default='PassiveAggressiveClassifier', help='Model to test data with (default=PasiveAggressiveClassifier')
  args = parser.parse_args() 

  _, classes, new_classes, X_train, X_test, y_train, y_test = process_data(args.file, args.df_words, args.df_tag, args.df_sentence)
  test_classifier(X_train, y_train, X_test, y_test, classes, new_classes, args.model) 
 

if __name__ == '__main__':
  main()
