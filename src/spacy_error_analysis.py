import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
import pandas as pd
import numpy as np
import argparse
pd.set_option('display.max_rows', 200)


######


"""

Error Analysis for trained spaCy model:

- load trained spaCy model
- load test data in .spacy format, transform back to "doc" object 
- transforms predicted doc and test doc into dataframes 
- displays differences between test set and predictions in the following format 
- highlights mistakes in the predictions in red


| word         | corr_label  | pred_label  |
|______________|_____________|_____________|
| jan	       |   B-PERS	 |    B-PERS   |
| van	       |   I-PERS	 |    I-PERS   |
| den	       |   I-PERS	 |    I-PERS   |
| keye	       |   I-PERS	 |    I-PERS   |
| vinhouters   |   B-PERS	 |    B-PERS   |
| zeghelsem	   |   B-LOC	 |    B-LOC    |
| onser	       |   B-PERS	 |     nan     |   --> nan will be highlighted in red
| vrouwen	   |   I-PERS	 |     nan     |   --> nan will be highlighted in red
| audenaerde   |   B-LOC	 |    B-PERS   |   --> B-PERS will be highlighted in red


print_scores() function can be used to print the scores for each label on the entire test set


#### TO RUN IN COMMAND LINE: #### 

- Pass required positional "trained_model" argument (= the path to "model-best")
- Pass required positional argument "test_set" (= the path to "training_data_TEST.spacy")
- Pass example number in test set to be processed



#### TO USE IN NOTEBOOK: ####

import spacy
from spacy_error_analysis import spacy_to_doc, print_scores, error_analysis

error_analysis(model_path, test_path, number)

__SECOND OPTION (faster when testing several examples):

from spacy_error_analysis import spacy_to_doc, print_scores, preds_vs_corr

nlp_ner = spacy.load(model_path)
test_set = spacy_to_doc(test_path)
preds_vs_corr(test_set, number, nlp_ner)

__PRINT_SCORES:

from spacy_error_analysis import spacy_to_doc, print_scores

nlp_ner = spacy.load(model_path)
test_set = spacy_to_doc(test_path)
print_scores(test_set, nlp_ner)

__RENDER_DISPLACY:

from spacy_error_analysis import spacy_to_doc, print_scores, render_displacy

nlp_ner = spacy.load(model_path)
test_set = spacy_to_doc(test_path)

render_displacy(test_set[number])   # for the correct version

text = test_set[number].text
doc = nlp_ner(text)

render_displacy(doc)  # for the model's prediction


Where:
- model_path = the path to your best model
- test_path = the path to your .spacy test set
- number = the example number in the test set you want to visualize (must be int type)


"""



def spacy_to_doc(path):
    """
    Loads .spacy data, turns it back into a list of Doc objects based on the spacy model that was used to turn it into a DocBin
    In this case, spacy.bank('nl')
    Returns test set as list of Doc objects
    """
    doc_bin = DocBin().from_disk(path)
    nlp = spacy.blank("nl")
    test_set = list(doc_bin.get_docs(nlp.vocab))

    return test_set



def print_scores(test_set, trained_model):
    """
    Loops over test set, makes predictions for all items with the trained model, puts both the "correct" test set Doc and the prediction Doc into an Example object
    Makes list of Example objects (including all items in test_set and their corresponding predictions)
    Returns scores of the trained model on test set
    """
    examples = []
    scorer = Scorer()
    for charter in test_set:
        text = charter.text
        pred = trained_model(text)
        example = Example(pred, charter)
        examples.append(example)
        
    return scorer.score(examples)


def preds_vs_corr(test_set, number, trained_model):
    """
    Allows for one example from test set to be chosen, makes predictions on that example with the trained model,
    and compares the result with the correct tags, highlighting the mistakes in the predictions
    """
    doc_text = test_set[number].text
    doc_doc = trained_model(doc_text)

    correct_list = []
    predicted_list = []
    text_list = []
    for predicted, correct in zip(doc_doc, test_set[number]):
        text_list.append(correct.text)
        correct_list.append(correct.ent_type_)
        predicted_list.append(predicted.ent_type_)

    list_of_tuples = list(zip(text_list, correct_list, predicted_list))
  
    df = pd.DataFrame(list_of_tuples, columns = ["word", "correct", "predicted"])
    df2 = df.replace(r'^\s*$', np.nan, regex=True)
    df2 = df2.dropna(subset=["correct", "predicted"], how='all')
    return df2.style.apply(lambda x: (x != df2['correct']).map({True: 'background-color: red; color: white', False: ''}), subset=['predicted'])



def error_analysis(path_model, path_test, number):
    """
    Combines previous functions:
    - loads trained model
    - loads and converts test data to list of Doc objects
    - Compares predictions and correct tags for chosen example, highlighting mistakes in predictions
    """
    nlp_ner = spacy.load(path_model)
    test_set = spacy_to_doc(path_test)
    return preds_vs_corr(test_set, number, nlp_ner)



def render_displacy(doc):
    """
    Particular to the middle-dutch dataset:
    - assigns a color to each tag
    - renders doc with chosen colors
    """
    colors = {"B-PERS": "#F67DE3", "I-PERS": "#F9C0F0", "B-LOC":"#26DDB1", "I-LOC":"#85F9DD", "B-DATE":"#427AF9", "I-DATE":"#ABC5FF", "B-MONEY":"#FFF11E", "I-MONEY":"#FCF7A8"}
    options = {"colors": colors} 
    return spacy.displacy.render(doc, style="ent", options= options, jupyter=True)



def main ():
    parser = argparse.ArgumentParser(prog = 'Tansform NER data into training data for SpaCy',
                    description = 'Process NER data into SpaCy trainin data, save SpaCy training data to file') 

    parser.add_argument('trained_model', type=argparse.FileType('r'), help="Dataset to be transformed into SpaCy training data")
    parser.add_argument('test_set', type=argparse.FileType('r'), help="Dataset to be transformed into SpaCy training data")
    parser.add_argument('example_number', type=int, help='an integer for the accumulator')

    args = parser.parse_args() 

    error_analysis(args.trained_model, args.test_set, args.example_number)

 
if __name__ == '__main__':
    main()
