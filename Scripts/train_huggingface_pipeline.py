import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier, Perceptron
from sklearn.preprocessing import FunctionTransformer
from sklearn.naive_bayes import MultinomialNB
import skops
from skops import hub_utils
from skops import card
from pathlib import Path
from transform_split_save_spacy import group_split
import argparse


######

"""
Trains NER model and saves it so it can be uploaded to Huggingface with Skops.
Saves X_test, y_test, new_classes (NER tags exluding 'O') and pipeline.



This script is a not a success. However, much of it could be recycled to be used with sentiment analysis intead. For NER, uploading models to Huggingface with Skops is not possible (in any decent way).

Problem description: 
1. MAIN PROBLEM: Skops only supports 4 kinds of tasks: tabular classification, tabular regression, text classification, text regression. Out of these 4, the closest task to NER is text classification. However, text classification
gives the entire text one label. This is good for sentiment analysis, but not for NER. NER requires token classificaiton, a task that is supported by Huggingface but not by Skops. 

2. In order to transform our data from text to tabular data in order to train our NER model, a custom function needed to be used. Huggingface does not have access to this custom function, so the model doesn't work on Huggingface. 
If this script is adapted to put a sentiment analysis model on Huggingface instead, this custom function will not be necessary, and therefore this will not be a problem.

3. Skops problem: This might be a problem that will come back even if a sentiment analysis model is uploaded with Skops: The example provided in the inference API on Huggingface for text classification is, by default, "I like you, I love you".
Even though skops provides the option to upload our own test data so it can be used in the inference API, this DOES NOT WORK for text classification (it DOES work for tabular classification). The default will be used regardless of the test data provided.

4. Skops problem: Very small problem, but for the model card, whatever information is written under "eval_results" will not appear on Huggingface (see push_to_huggingface script). This is the case for every single other mmodel
uploaded to Huggingface via skops.  


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


"""



def load_process(path,  words="word", tag="tag", sentence="sentence", random_state=42):
    """
    We want our data to appear as a text in Huggingface, as opposed to one word per column. Therefore we group our data by sentence (charter in this case). 
    This data will later need to be converted back to a tabular format so the model can be trained on it.

    Load data, group by sentence, train-test split
    Convert y_train and y_test back to a flat list (instead of grouped by sentence)
    This cannot be done with X_train and X_test at this point because we want the sentences to appear as sentences on the Huggingface inference API. 
    So the data conversion from text back to tabular must be done within the pipeline, that way it will also be done to the data on the inference API on Huggingface

    Returns X_train and X_test grouped by sentence, and y_train and y_test as the format we need to train the model (list of tags instead of list of list of tags), 
    since the y-data doesn't get processed in the pipeline
    """
    df = pd.read_csv(path)
    X, y, classes, new_classes = group_split(df, sentence, words, tag)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=random_state)


    # Revert y_train and y_test but not X 
    flat_y = [item for sublist in y_train for item in sublist]
    df = pd.DataFrame(flat_y)
    df.columns = ['tag']
    y_train = df['tag'].values

    flat_y = [item for sublist in y_test for item in sublist]
    df = pd.DataFrame(flat_y)
    df.columns = ['tag']
    y_test = df['tag'].values

    return X_train, X_test, y_train, y_test, classes, new_classes



def revert_data(X):
    """
    Function to be used within the pipeline to transform the X-data into the format we need to train the model
    Splits sentences back into words and puts them into dataframe, makes list from dataframe column containing words (list of words instead of list of list of words)
    Output X_train and X_test should be same length as y_train and y_test respectively, evidently
    """
    words = [i.split() for i in X]
    flat_X = [item for sublist in words for item in sublist]
    df = pd.DataFrame(flat_X)
    df.columns = ['word']
    X = df['word']

    return X
    

def train_model(name, X_train, y_train, random_state=42, *args, **kwargs):
    """
    Trains NER model
    Choice between several models:  PassiveAggressiveClassifier, SGDClassifier, Perceptron and MulinomialNB, other models could be used as well
    Puts our previously defined revert_data function into a FunctionTransformer() so it can be used within the pipeline to transform our X-data before training
    Trains pipeline, returns pipeline


    Note: This is a problem in Huggingface, as Hugginface doesn't have access to our custom function. However, this script is not meant to be used for NER in the end anyways,
    so this part will not be necessary. If used for sentiment analysis, this custom function can be dropped and it would work normally.

    """
    # Train model
    try:
        classifier = name(*args, **kwargs, random_state=random_state)
    except TypeError: # some classifiers don't have random_state
        classifier = name(*args, **kwargs)

    transformer = FunctionTransformer(revert_data)

    pipeline = Pipeline([
    ("trans", transformer),
    ('vectorizer', CountVectorizer()),
    ('classifier', classifier)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline



def save_model(pipeline, filename):
    """
    Save pipeline with pickle
    """
    with open(filename, mode="bw") as f:
        pickle.dump(pipeline, file=f)


######


def main():
    CONVERSION_TABLE = {'Perceptron': Perceptron,
                    'SGDClassifier': SGDClassifier,
                    'NultinomialNB': MultinomialNB,
                    'PassiveAggressiveClassifier': PassiveAggressiveClassifier
                    }

    class RenameOption(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
         setattr(namespace, self.dest, CONVERSION_TABLE[values])


    parser = argparse.ArgumentParser(prog = 'Train classifier on NER data and push to Huggingface',
                    description = 'Process NER data,') 

    parser.add_argument('file', type=argparse.FileType('r'), help="Dataset to be transformed into SpaCy training data")

    parser.add_argument("-fn", "--model_filename", type=str, default="middle_dutch_passAgg", help="Filename for model (default: %(default)s)")
    parser.add_argument('-m', '--model', action=RenameOption, nargs='?', choices=CONVERSION_TABLE, default=PassiveAggressiveClassifier, 
    const=PassiveAggressiveClassifier, help='Model to test data with (default: %(default)s)')
    parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default: %(default)s)")
    parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default: %(default)s)")
    parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default: %(default)s)")
    cmd_args = parser.parse_args() 
    print(cmd_args)


    X_train, X_test, y_train, y_test, _, new_classes = load_process(cmd_args.file, cmd_args.df_words, cmd_args.df_tag, cmd_args.df_sentence)

    print(len(X_train))
    print(len(y_train))

    # X_test.to_csv("X_test_huggingface.csv", index=False)
    save_model(X_test, "X_test_huggingface")
    np.savez_compressed("y_test_huggingface.ar", y_test)
    save_model(new_classes, "new_classes")

    pipeline = train_model(cmd_args.model, X_train, y_train, random_state=42)
    save_model(pipeline, cmd_args.model_filename)


if __name__ == '__main__':
    main()
