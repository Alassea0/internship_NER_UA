import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay, precision_recall_fscore_support
import sklearn
import pickle
from sklearn.linear_model import PassiveAggressiveClassifier
import skops
from skops import hub_utils
import matplotlib.pyplot as plt
from skops import card
from pathlib import Path


def load_process_train(path, name, random_state=42, *args, **kwargs):
    """
    Load csv, split in X-y, vectorize X, get list of classes excluding "O", train-test split, train model
    Return trained model, new_classes list, X_test and y_test
    """
    # Load data
    df = pd.read_csv(path)
    df['sentence'] = df['sentence'].astype(str)
    
    # X-y split, vectorize X
    X = df.drop(['tag'], axis=1)
    v = DictVectorizer(sparse=False)
    X = v.fit_transform(X.to_dict('records'))
    y = df.tag.values
    
    # New classes = classes minus "O" class, which is most of the data
    classes = np.unique(y)
    classes = classes.tolist()
    new_classes = classes.copy()
    new_classes.pop()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10, random_state=42)
    
    # Train model
    try:
        classifier = name(*args, **kwargs, random_state=random_state)
    except TypeError: # some classifiers don't have random_state
        classifier = name(*args, **kwargs)

    classifier.partial_fit(X_train, y_train, classes)

    return classifier, new_classes, X_test, y_test


def save_model(model, filename):
    """
    Save model with pickle
    """
    with open(filename, mode="bw") as f:
        pickle.dump(model, file=f)


def create_repo(local_repo, X_test):
    """
    Create repository on Hub, except if it already exists
    """
    try:
        hub_utils.init(
        model=model_filename,
        requirements=[f"scikit-learn={sklearn.__version__}"],
        dst=local_repo,
        task="tabular-classification",
        data=X_test,
        )
    except OSError: 
        pass


def make_model_card(local_repo, model, X_test, y_test, new_classes):
    """
    Make model card for Hugginface, including confusion matrix and accuracy/precision/recall/f1 scores
    """
    # Initiate model card, adds metadata
    model_card = card.Card(model, metadata=card.metadata_from_config(Path(local_repo)))

    # Define model card contents
    citation_bibtex = "**BibTeX**\n\n```\n@inproceedings{...,year={2022}}\n```"
    authors = "Alassea TEST"
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



def model_to_huggingface(path, model, repo_id, model_filename):
    """
    Combination of all previous functions + push to Huggingface Hub
    """
    model, new_classes, X_test, y_test = load_process_train(path, model)
    save_model(model, model_filename)
    local_repo = "model_local_repo"
    create_repo(local_repo, X_test)
    make_model_card(local_repo, model, X_test, y_test, new_classes)
    # If the repository doesn't exist remotely on the Hugging Face Hub, it will be created when we set create_remote to True
    hub_utils.push(
        repo_id=repo_id,
        source=local_repo,
        token="", # personal token to be downloaded from huggingface
        commit_message="Sixth attempt PassiveAggressive middle dutch NER",
        create_remote=True,
        )


#path = "/home/ada/Documents/DTA/Internship/Task_1/C14NL.csv"
path_git = "./Data/C14NL.csv"
passAgg = PassiveAggressiveClassifier
repo_id = "Alassea/middle-dutch-NER_passAgg"
model_filename = "middle_dutch_passAgg.pkl"

model_to_huggingface(path_git, passAgg, repo_id, model_filename)

