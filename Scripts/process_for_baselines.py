import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import argparse


######  

"""
Tests baselines for a NER dataset:
  - Vectorizes data
  - train-test split
  - trains classifier (choice between 4 classifiers: Perceptron, MultinomialNB, SGDClassifier, PassiveAggressiveClassifier)
  - prints classification report


Requires a .csv file in the following data format:
 
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
- If the names of the columns are NOT word, tag and sentence, pass:
    - "-sen your_sentence_colname" for the sentence column
    - "-w your_word_colname" for the word column
    - "-t your_tag_colname" for the tag column
- Default classifier is PassiveAggressiveClassifier, if you want to test another one, pass:
    - "-m Perceptron" for Perceptron, replace with desires classifier name for the other 2 



TO USE IN NOTEBOOK:

_, classes, new_classes, X_train, X_test, y_train, y_test = process_data(file_path, df_words_colname, df_tag_colname, df_sentence_colname)
test_classifier(X_train, y_train, X_test, y_test, classes, new_classes, model_choice) 

Where:
- file_path = the path to your data file
- df_words_colname, df_tag_colname and df_sentence_colname are the names of the "word", "tag" and "sentence" column names in your data. These are set to "word", "tag" and "sentence" by default
- model_choice = Perceptron, MultinomialNB or SGDClassifier. Is set to PassiveAggressiveClassifier by default.

"""


 
def remove_cols(path, words="word", tag="tag", sentence="sentence"):
    """
    Function particular to the Middle-dutch dataset, used separately to remove unnecessary columns
    """
    df = pd.read_csv(path,  sep='\t', names=[words, tag, '365', 'X', sentence])
    df = df.drop(['365', 'X'], axis=1)

    return df


def process_data(path, words="word", tag="tag", sentence="sentence"):
    """
    Vectorize data, put data in X, tags in y, extract classes and new classes (classes minus "O")
    Train-test split
    """
    # Load data
    df = pd.read_csv(path)
    df[sentence] = df[sentence].astype('str')
    
    # X-y split, vectorize X
    X = df.drop([tag], axis=1)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    y = df.tag.values
    
    # New classes = classes minus "O" class, which is most of the data
    classes = np.unique(y)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()


    classes = np.unique(y)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=0)

    return df, classes, new_classes, X_train, X_test, y_train, y_test


def test_classifier(X_train, y_train, X_test, y_test, classes, new_classes, name, random_state=42, *args, **kwargs):
    """
    Prints classification report 
    Allows for different models to be tested, with their arguments
    """
    try:
      classifier = name(*args, **kwargs, random_state=random_state)
    except TypeError: # some classifiers don't have random_state
      classifier = name(*args, **kwargs)
    print(classifier) # print classifier parameters
    classifier.partial_fit(X_train, y_train, classes)
    print(classification_report(y_pred=classifier.predict(X_test), y_true=y_test, labels=new_classes))



def main():
    CONVERSION_TABLE = {'Perceptron': Perceptron,
                      'SGDClassifier': SGDClassifier,
                      'MultinomialNB': MultinomialNB,
                      'PassiveAggressiveClassifier': PassiveAggressiveClassifier
                      }

    class RenameOption(argparse.Action):
      def __call__(self, parser, namespace, values, option_string=None):
          setattr(namespace, self.dest, CONVERSION_TABLE[values])

    parser = argparse.ArgumentParser(prog = 'Process for baselines',
                    description = 'Processes NER data and tests baseline performances with PassiveAggressiveClassifier.' 
                    'Allows for testing with 3 additional classifiers: Perceptron, SGDClassifier, MulinomialNB') 

    parser.add_argument('file', type=argparse.FileType('r'), help="Path to NER dataset to be processed for baselines")

    parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default: %(default)s)")
    parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default: %(default)s)")
    parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default: %(default)s)")
    parser.add_argument('-m', '--model', action=RenameOption, nargs='?', choices=CONVERSION_TABLE, default=PassiveAggressiveClassifier, 
    const=PassiveAggressiveClassifier, help='Model to test data with (default: %(default)s)')
    args = parser.parse_args() 

    _, classes, new_classes, X_train, X_test, y_train, y_test = process_data(args.file, args.df_words, args.df_tag, args.df_sentence)
    test_classifier(X_train, y_train, X_test, y_test, classes, new_classes, args.model) 
 

if __name__ == '__main__':
    main()
