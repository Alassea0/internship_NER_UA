import sklearn_crfsuite
from sklearn.model_selection import train_test_split
from sklearn_crfsuite.utils import flatten
from functools import wraps
import argparse
from process_for_baselines import process_data
import pickle


###### 


"""

Trains CRF suite model on NER data, saves model and prints classification report


Requires a .csv file in the following data format, column names MUST BE "word", "tag" and "sentence":
 
| word      |  tag    | sentence |
|___________|_________|__________|
| The       |   'O'   |    1     |
| Uyuni     | 'B-LOC' |    1     |
| salt      |   'O'   |    1     |
| flats     |   'O'   |    1     |
| are       |   'O'   |    1     |
| in        |   'O'   |    1     |
| Bolivia   | 'B-LOC' |    1     |
| When      |   'O'   |    2     |
| it        |   'O'   |    2     |
| rains     |   'O'   |    2     |



TO RUN IN COMMAND LINE: 

- Pass required positional "file" argument (= the path to the data file)
- Model name is "CRF_suite_trained" by default, if you want another name, pass:
    - "-fn your_model_name"



TO USE IN NOTEBOOK:

from process_for_baselines import process_data

df, _, new_classes, _, _, _, _ = process_data(file_path)
crf, _, X_test, _, y_test = train_crf(df)
save_model(crf, model_filename)
y_pred = crf.predict(X_test)
print(flat_classification_report(y_test, y_pred, labels = new_classes))

Where:
- file_path = the path to your data file
- model_filename = the name for the saved model

"""


class SentenceGetter(object):
    
    def __init__(self, data, word="word", tag="tag", sentence="sentence"):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s[word].values.tolist(),  
                                                           s[tag].values.tolist())]
        self.grouped = self.data.groupby(sentence).apply(agg_func)
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


## code copy-pasted from https://github.com/MeMartijn/updated-sklearn-crfsuite.git#egg=sklearn_crfsuite #############################################################
## This fixes the "TypeError: classification_report() takes 2 positional arguments but 3 were given" error when trying to use flat_classification_report from metrics

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


######################################################################################################################################################################


def train_crf(df):
    """
    Processes data and trains CRF suite model
    """
    getter = SentenceGetter(df)
    sentences = getter.sentences
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

    crf = sklearn_crfsuite.CRF(
        algorithm='pa',
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    return crf, X_train, X_test, y_train, y_test


def save_model(model, filename):
    """
    Save model with pickle
    """
    with open(filename, mode="bw") as f:
        pickle.dump(model, file=f)


def main():
    parser = argparse.ArgumentParser(prog = 'Train NER data with sklearn-crfsuite',
                    description = 'Process NER data and train sklearn-crfsuite model on it. Print classification report.') 

    parser.add_argument('file', type=argparse.FileType('r'), help="Path to dataset to be processed for baselines")

    parser.add_argument("-fn", "--model_filename", type=str, default="CRF_suite_trained", help="Filename for model (default: %(default)s)")
    args = parser.parse_args() 

    df, _, new_classes, _, _, _, _ = process_data(args.file)
    crf, _, X_test, _, y_test = train_crf(df)
    save_model(crf, args.model_filename)

    y_pred = crf.predict(X_test)
    print(flat_classification_report(y_test, y_pred, labels = new_classes))



if __name__ == '__main__':
    main()