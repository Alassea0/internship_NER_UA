import pandas as pd
import numpy as np
from string import punctuation
from sklearn.model_selection import train_test_split
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
import argparse


 
###### 

"""

Transforms NER training data into format that can be used to train spaCy. 
Splits into train, test and dev sets and saves all three.

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

Transforms it into the following format:

{'text': 'the uyuni salt flats are in bolivia', 'entities': [(4, 9, 'B-LOC'), (29, 36, 'B-LOC')]}

And then transforms this into a spacy DocBin object.


TO RUN IN COMMAND LINE: 

- Pass required positional "file" argument (= the path to the data file)
- If the names of the columns are not word, tag and sentence, pass:
    - "-sen your_sentence_colname" for the sentence column
    - "-w your_word_colname" for the word column
    - "-t your_tag_colname" for the tag column
- The training data output is named "training_data" by default (training_data_TRAIN, training_data_TEST and training_data_DEV). If you want anothe filename, pass:
    - "-fn your_filename"


TO USE IN NOTEBOOK:

from transform_split_save_spacy import make_train_data, training_to_docbin

train_data, test_data, dev_data = make_train_data(file_path, df_sentence_colname, df_word_colname, df_tag_colname)
## OR, if default column names are used:
train_data, test_data, dev_data = make_train_data(file_path)

train_docbin = training_to_docbin(train_data)
test_docbin = training_to_docbin(test_data)
dev_docbin = training_to_docbin(dev_data)

train_docbin.to_disk(data_filename + "_TRAIN.spacy")
test_docbin.to_disk(data_filename + "_TEST.spacy")
dev_docbin.to_disk(data_filename + "_DEV.spacy")

Where:
- file_path = the path to your data file
- df_words_colname, df_tag_colname and df_sentence_colname are the names of the "word", "tag" and "sentence" column names in your data. These are set to "word", "tag" and "sentence" by default
- data_filename = the name for the saved training data



"""


def clean_words(word):
    """
    Lowers text and removes punctuation
    """
    word = word.casefold()
    word = word.translate(str.maketrans('', '', punctuation))

    return word



def group_split(df, sentence="sentence", words="word", tag="tag"):
    """
    Group df by sentence, save text in X and tags in y, 
    define classes and new_classes (= classes excluding the "O" tag, which is most of the data)

    If your data has 500 sentences and 10.000 words, the length of X and y is now 500, not 10.000
    """
    body = []
    tags = []

    grouped = df.groupby(sentence)

    for group in grouped.groups:
        df = grouped.get_group(group)
        body.append(' '.join(list(df[words])))
        tags.append(list(df[tag]))

    X = body
    y = tags

    labels = df.tag.values
    classes = np.unique(labels)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()

    return X, y, classes, new_classes


def train_test_split_data(X, y, random_state=42):
    """
    Train-test-dev split 80-10-10
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)
    X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size = 0.5, random_state=random_state)

    return X_train, y_train, X_test, y_test, X_dev, y_dev


def convert_data(X, y, new_classes):
    """
    Converts data to SpaCy training data:
    - Counts start and end token for entities that are not 'O'
    - Makes dictionary with the new_classes and annotations:
        - annotations include text and entities:
            * text = X, the words grouped by sentence
            * entities = start and end token of the entity + the entity
    """
    nlp = spacy.load('nl_core_news_sm')

    # Initiate dictionary, classes = new_classes (labels excluding 'O')
    training_data = {'classes' : new_classes, 'annotations' : []}

    # Loop over X and y, initiate second dictionary, assign X to text
    for sentence, entity in zip(X, y):
      temp_dict = {}
      temp_dict['text'] = sentence
      temp_dict['entities'] = []

      # Initiate empty start and stop lists
      start = []
      stop = []
      # Put X (sentence) throuh spacy's nl_core_news_sm model so ce can extract the token indexes 
      doc = nlp(sentence)
      for token in doc:
          start_ = token.idx # for the start token, we can simply used the token's idx provided by spaCy
          stop_ = ((token.idx + len(token.text))) # for the end token, we take the token's idx + the length of the token
          start.append(start_)
          stop.append(stop_)

      # Loop over the start and stop lists with the indexes of each token, as well as entity (y)
      zippity = zip(start, stop, entity)
      for start_, stop_, ent in zippity:
        # Ignore entity if it's tag is 'O', we only need the start and end token of the actual entities
        if ent == "O":
          pass
        # If the entity's tag us not 'O', append its start and end tokens, as well as the entity itself to the dictionary 
        else:
          temp_dict['entities'].append((start_, stop_, ent))
      training_data['annotations'].append(temp_dict)

    return training_data


def make_train_data(path, sentence="sentence", words="word", tag="tag"):
    """
    Combines previous functions to make training data from .csv file in the correct format:
    - Reads csv
    - Changes the sentence column to str type instead of int
    - Cleans text
    - Groups by sentence and splits into X, y
    - Train-test-dev splits X and y
    - Converts train, test and dev
    - Returns converted train, test and dev data (dictionary)
    """
    df = pd.read_csv(path)
    df[sentence] = df[sentence].astype('str')
    df[words] = df[words].apply(clean_words)
    X, y, _, new_classes = group_split(df, sentence, words, tag)
    X_train, y_train, X_test, y_test, X_dev, y_dev = train_test_split_data(X, y)

    training_data = convert_data(X_train, y_train, new_classes)
    test_data = convert_data(X_test, y_test, new_classes)
    dev_data = convert_data(X_dev, y_dev, new_classes)

    return training_data, test_data, dev_data


def training_to_docbin(training_data):
    """
    Converts training data from the dictionary format into spaCy DocBin object
    """
    nlp = spacy.blank("nl") 
    doc_bin = DocBin() 

    for training_example in tqdm(training_data['annotations']): 
        text = training_example['text']
        labels = training_example['entities']
        doc = nlp.make_doc(text) 
        ents = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        filtered_ents = filter_spans(ents)
        print(filtered_ents)
        doc.ents = filtered_ents 
        doc_bin.add(doc)

    return doc_bin


def main ():
    parser = argparse.ArgumentParser(prog = 'Tansform NER data into training data for SpaCy',
                    description = 'Process NER data into SpaCy trainin data, save SpaCy training data to file') 

    parser.add_argument('file', type=argparse.FileType('r'), help="Dataset to be transformed into SpaCy training data")

    parser.add_argument("-fn", "--data_filename", type=str, default="training_data", help="Filename for training data (default: %(default)s)")
    parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default: %(default)s)")
    parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default: %(default)s)")
    parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default: %(default)s)")
    args = parser.parse_args() 

    train_data, test_data, dev_data = make_train_data(args.file, args.df_sentence, args.df_words, args.df_tag)

    print(len(train_data['annotations']))
    print(len(test_data['annotations']))
    print(len(dev_data['annotations']))

    train_docbin = training_to_docbin(train_data)
    test_docbin = training_to_docbin(test_data)
    dev_docbin = training_to_docbin(dev_data)

    train_docbin.to_disk(args.data_filename + "_TRAIN.spacy")
    test_docbin.to_disk(args.data_filename + "_TEST.spacy")
    dev_docbin.to_disk(args.data_filename + "_DEV.spacy")
 
if __name__ == '__main__':
    main()

