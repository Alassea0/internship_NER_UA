import pandas as pd
import numpy as np
from string import punctuation
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
from spacy.util import filter_spans
import argparse
import os


parser = argparse.ArgumentParser() 
parser.add_argument('-f', '--file', type=argparse.FileType('r'), help="Dataset to be transformed into SpaCy training data")
parser.add_argument("-fn", "--data_filename", type=str, default="training_data", help="Filename for training data")
parser.add_argument("-sen", "--df_sentence", type=str, default="sentence", help="Column name for sentences (default ='sentence')")
parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default ='word')")
parser.add_argument("-t", "--df_tag", type=str, default="tag", help="Column name for tags (default = 'tag')")
args = parser.parse_args() 


######



def clean_words(word):
  word = word.casefold()
  word = word.translate(str.maketrans('', '', punctuation))
  return word



def group_split(df):
    """
    Group df by sentence, save text in X and tags in y, define classes 
    """
    body = []
    tags = []

    grouped = df.groupby(args.df_sentence)

    for group in grouped.groups:
        df = grouped.get_group(group)
        body.append(' '.join(list(df[args.df_words])))
        tags.append(list(df[args.df_tag]))

    X = body
    y = tags

    labels = df.tag.values
    classes = np.unique(labels)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()

    return X, y, new_classes


def convert_data(X, y, new_classes):
    """
    Converts data to SpaCy training data:
    - Puts all text together
    - Counts start and end token for entities that are not 'O'
    """
    nlp = spacy.load('nl_core_news_sm')

    training_data = {'classes' : new_classes, 'annotations' : []}

    for sentence, entity in zip(X, y):
      temp_dict = {}
      temp_dict['text'] = sentence
      temp_dict['entities'] = []

      start = []
      stop = []
      doc = nlp(sentence)
      for token in doc:
          start_ = token.idx
          stop_ = ((token.idx + len(token.text))) 
          start.append(start_)
          stop.append(stop_)

      zippity = zip(start, stop, entity)
      for start_, stop_, ent in zippity:
        if ent == "O":
          pass
        else:
          temp_dict['entities'].append((start_, stop_, ent))
      training_data['annotations'].append(temp_dict)

    return training_data



df = pd.read_csv(args.file)
df[args.df_sentence] = df[args.df_sentence].astype('str')
df[args.df_words] = df[args.df_words].apply(clean_words)
X, y, new_classes = group_split(df)
training_data = convert_data(X, y, new_classes)

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


doc_bin.to_disk(args.data_filename + ".spacy")