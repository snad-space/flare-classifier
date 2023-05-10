import os
import json

from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

EVAL_PATH = "metrics"


def evaluate_metrics(model, true_labels, predict_labels, split):
    recall = recall_score(true_labels, predict_labels)
    precision = precision_score(true_labels, predict_labels)
    accuracy = accuracy_score(true_labels, predict_labels)
    fscore = f1_score(true_labels, predict_labels)

    prc_dir = os.path.join(EVAL_PATH, model)
    os.makedirs(prc_dir, exist_ok=True)
    prc_file = os.path.join(prc_dir, f"{split}_metrics.json")

    with open(prc_file, "w") as fd:
        json.dump({"prc": {"precision": precision, "recall": recall, "accuracy": accuracy, "f1_score": fscore}}, fd)
