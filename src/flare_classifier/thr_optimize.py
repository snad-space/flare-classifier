import numpy as np
from sklearn.metrics import fbeta_score


def optimal_threshold(y_probs, true_labels):
    """
    Function to define a threshold which maximizie f-beta score (beta=0.3)
    """

    thrs = np.arange(0, 1, 0.001)
    scores = []

    for t in thrs:
        y_labels = (y_probs > t).astype("int")
        fscore = fbeta_score(true_labels, y_labels, beta=0.3)
        scores.append(fscore)

    optimal_thr = thrs[np.argmax(scores)]

    return optimal_thr
