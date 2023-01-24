import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay, precision_recall_fscore_support
import sklearn
import pickle
import skops
from skops import hub_utils
from skops import card
from pathlib import Path
import argparse 

######

"""
Loads NER model, X_test, y_test and new_classes, uploads it to Huggingface with Skops.


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




def load_model(model_path):
    """
    Loads model with pickle
    """
    with open(model_path, "rb") as input_file:
        model = pickle.load(input_file)
    
    return model

def load_array(path):
    """
    Loads array with numpy
    """
    y_test = np.load(str(path), allow_pickle=True)["arr_0"]

    return y_test


def create_repo(local_repo, X_test, model, model_filename):
    """
    Create repository on Hub, except if it already exists

    Note 1: If it already exists, no changes will be saved if this code is run again. If you wish to make changes to this part after initializing it (for example, by changing it form text-classification 
    to tabular-classification), the repo will first need to be deleted, or given a new name

    Note 2: the X_test data provided here will be the example used in the Huggingface inference API for tabular classification but NOT for text-classification. For text-classification, the default "I like you, I love you"
    example will be used regardless of the test data provided here and regardless of the language of the model. This seems to be a problem with skops. 

    """
    model_filename = model_filename + ".pkl"
    with open(model_filename, mode="bw") as f:
        pickle.dump(model, file=f)

    try:
        hub_utils.init(
        model=model_filename,
        requirements=[f"scikit-learn={sklearn.__version__}"],
        dst=local_repo,
        task="text-classification",
        data=X_test,
        )
    except OSError: 
        pass



def make_model_card(local_repo, model, X_test, y_test, new_classes):
    """
    Make model card for Hugginface, including confusion matrix and accuracy/precision/recall/f1 scores


    Note: eval_results does NOT APPEAR on Huggingface. This seems to be a problem with Skops, the rest does appear. 
    """
    # Initiate model card, adds metadata
    model_card = card.Card(model, metadata=card.metadata_from_config(Path(local_repo)))

    # Define model card contents
    citation_bibtex = "**BibTeX**\n\n```\n@inproceedings{...,year={2022}}\n```"
    authors = "TEST"
    description = "Middle Dutch NER with PassiveAgressiveClassifier"
    limitations = "This model is not ready to be used in production."
    training_procedure = "TESTING"
    eval_results = "The model is evaluated on test data using accuracy and F1-score with macro average. TEST! THIS, JUST THIS, DOES NOT GET ADDED TO THE MODEL CARD"

    # You can add several sections and metadata to the model card
    model_card.add(
        **{"Citation": citation_bibtex,
        "Model Card Authors": authors,
        "Model description": description,
        "Model description/Intended uses & limitations": limitations,
        "Model description/Training Procedure": training_procedure,
        "Model description/Evaluation Results": eval_results,
        })


    y_pred = model.predict(X_test)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=new_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=new_classes)
    disp.plot()
    disp.figure_.savefig(Path(local_repo) / "confusion_matrix.png")

    model_card.add_plot(**{"Model description/Evaluation Results/Confusion Matrix": "confusion_matrix.png"})
    
    # Accuracy, precision, recall and f1 scores with and without the "O" class
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="micro")
    scores = precision_recall_fscore_support(y_pred=y_pred, y_true=y_test, average='micro', labels=new_classes)
    precision_micro = scores[0]
    recall_micro = scores[1]
    f1_micro = scores[2]

    model_card.add_metrics(**{"accuracy including 'O'": accuracy, "f1 score including 'O": f1, "precision excluding 'O'": precision_micro, "recall excluding 'O'": recall_micro, "f1 excluding 'O'": f1_micro})
    
    # Save model card locally
    model_card.save(Path(local_repo) / "README.md")




def model_to_huggingface(X_test, y_test, new_classes, model, model_filename, repo_id, commit_message, token):
    """
    Combination of all previous functions + push to Huggingface Hub
    """

    local_repo = "model_local_repo"
    create_repo(local_repo, X_test, model, model_filename)
    make_model_card(local_repo, model, X_test, y_test, new_classes)
    # If the repository doesn't exist remotely on the Hugging Face Hub, it will be created when we set create_remote to True
    hub_utils.push(
        repo_id=repo_id,
        source=local_repo,
        token=token, # personal token to be downloaded from huggingface
        commit_message=commit_message,
        create_remote=True,
        )



######

def main():
    parser = argparse.ArgumentParser(prog = 'Push trained NER model to Huggingface',
                    description = 'Load pickled trained NER model and other necessary files, push to Huggingface') 

    parser.add_argument('trained_model', help="Path to trained model file")

    parser.add_argument('X_test', help="Path to X_test fie (.csv)")
    parser.add_argument('y_test', help="Path to y_test fie (np.array)")
    parser.add_argument('new_classes', help="Path to new_classes file (pickled list)")
    parser.add_argument("token", type=str, help="User authentication token form Huggingface")

    parser.add_argument("-fn", "--model_filename", type=str, default="middle_dutch_text-classification", help="Filename for model (default: %(default)s)")
    parser.add_argument("-repo", "--repo_id", type=str, default="Alassea/middle-dutch-NER_text-classification", help="Huggingface repo ID (default: %(default)s)")
    parser.add_argument("-cm", "--commit_message", type=str, default="Update model", help="Commit message for Huggingface (default: %(default)s)")
    parser.add_argument("-w", "--df_words", type=str, default="word", help="Column name for words (default: %(default)s)")
    cmd_args = parser.parse_args() 

    pipeline = load_model(cmd_args.trained_model)
    # X_test = pd.read_csv(cmd_args.X_test)
    # X_test = X_test[cmd_args.df_words]
    y_test = load_array(cmd_args.y_test)
    X_test = load_model(cmd_args.X_test)
    new_classes = load_model(cmd_args.new_classes)


    model_to_huggingface(X_test, y_test, new_classes, pipeline, cmd_args.model_filename, cmd_args.repo_id, cmd_args.commit_message, cmd_args.token)


if __name__ == '__main__':
    main()
