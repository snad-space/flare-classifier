import os
import json

import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, accuracy_score, fbeta_score, auc, roc_curve

EVAL_PATH = "metrics"


def evaluate_metrics(model, true_labels, predict_labels, split):
    recall = recall_score(true_labels, predict_labels)
    precision = precision_score(true_labels, predict_labels)
    accuracy = accuracy_score(true_labels, predict_labels)
    fscore = fbeta_score(true_labels, predict_labels, beta=0.3)

    prc_dir = os.path.join(EVAL_PATH, model)
    os.makedirs(prc_dir, exist_ok=True)
    prc_file = os.path.join(prc_dir, f"{split}_metrics.json")

    with open(prc_file, "w") as fd:
        json.dump({"prc": {"precision": precision, "recall": recall, "accuracy": accuracy, "fb_score": fscore}}, fd)


def evaluate_real(model, true_labels, predict_proba):
    fpr, tpr, thrs = roc_curve(true_labels, predict_proba, drop_intermediate=False)
    score = auc(fpr, tpr)

    prc_dir = os.path.join(EVAL_PATH, model)
    os.makedirs(prc_dir, exist_ok=True)
    prc_file = os.path.join(prc_dir, f"real_auc.json")

    with open(prc_file, "w") as fd:
        json.dump({"prc": {"auc": score, "fpr": fpr.tolist(), "tpr": tpr.tolist(), "thr": thrs.tolist()}}, fd)

    plt.plot(fpr, tpr, label=f"AUC = {score:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.title(model)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(prc_dir, "roc_curve.png"))
