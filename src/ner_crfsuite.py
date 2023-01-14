import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.utils import flatten
from functools import wraps
import pandas as pd
import argparse
from process_for_baselines import process_data



###### 


class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s['word'].values.tolist(),  
                                                           s['tag'].values.tolist())]
        self.grouped = self.data.groupby('sentence').apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
    def get_next(self):
        try: 
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s 
        except:
            return None


def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0, 
        'word.lower()': word.lower(), 
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]
def sent2labels(sent):
    return [label for token, label in sent]
def sent2tokens(sent):
    return [token for token, label in sent]


def _flattens_y(func):
    @wraps(func)
    def wrapper(y_true, y_pred, *args, **kwargs):
        y_true_flat = flatten(y_true)
        y_pred_flat = flatten(y_pred)
        return func(y_true_flat, y_pred_flat, *args, **kwargs)
    return wrapper


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):
    """
    Return classification report for sequence items.
    """
    from sklearn import metrics
    return metrics.classification_report(y_true, y_pred, labels=labels, **kwargs)



def train_crf(df):
    getter = SentenceGetter(df)
    sentences = getter.sentences
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    return crf, X_train, X_test, y_train, y_test


def main ():
    parser = argparse.ArgumentParser() 
    parser.add_argument('-f', '--file', type=argparse.FileType('r'), help="Path to dataset to be processed for baselines")
    parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default ='sentence')")
    parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default ='word')")
    parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default = 'tag')")
    args = parser.parse_args() 

    df, _, new_classes, _, _, _, _ = process_data(args.file)
    crf, X_train, X_test, y_train, y_test = train_crf(df)
    y_pred = crf.predict(X_test)
    print(flat_classification_report(y_test, y_pred, labels = new_classes))




if __name__ == '__main__':
    main()