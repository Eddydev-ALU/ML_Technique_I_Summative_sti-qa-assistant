"""
evaluate.py - Metrics shared by every model, so all experiments are scored the same way.

Every model produces a SCORE MATRIX: one row per question, one column per intent.
A higher score means "this intent is more likely". From the matrix we compute:

  accuracy@1 : the top-ranked intent is the correct one
  recall@k   : the correct intent is somewhere in the top k (k = 3, 5)
  MRR        : mean of 1 / (rank of the correct intent); 1.0 = always first, 0.5 = always second
  macro-F1   : F1 of the top-1 prediction per intent, averaged so every intent counts equally
               (small intents matter as much as big ones)
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RESULTS_FILE = os.path.join("results", "experiments.csv")


def true_ranks(scores, true_idx):
    """Rank (1 = best) of the correct intent for each question.

    Ties are counted pessimistically: if the correct intent is tied with others, it gets
    the worst of the tied positions. Otherwise a model that gives every intent a score of 0
    (e.g. BM25 when no word matches) would look perfect.
    """
    true_scores = scores[np.arange(len(true_idx)), true_idx]
    return (scores >= true_scores[:, None]).sum(axis=1)


def ranking_metrics(scores, true_idx, ks=(3, 5)):
    """Return a dict of metrics for one score matrix and the correct intent indices."""
    scores = np.asarray(scores, dtype=float)
    true_idx = np.asarray(true_idx)
    ranks = true_ranks(scores, true_idx)
    top1 = scores.argmax(axis=1)

    metrics = {"n": len(true_idx), "accuracy@1": float(np.mean(ranks == 1))}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    # Average over the intents that appear in this evaluation set
    metrics["macro_f1"] = float(f1_score(true_idx, top1, labels=np.unique(true_idx),
                                         average="macro", zero_division=0))
    return metrics


def per_category_accuracy(scores, true_idx, categories):
    """Accuracy@1 for each category (HIV, Syphilis, ...), to find weak areas."""
    correct = np.asarray(scores).argmax(axis=1) == np.asarray(true_idx)
    table = pd.DataFrame({"category": categories, "correct": correct})
    return table.groupby("category")["correct"].agg(n="size", accuracy="mean").round(3)


def save_result(experiment, split, metrics, notes=""):
    """Add (or replace) one row in results/experiments.csv, keyed by experiment + split."""
    row = {"experiment": experiment, "split": split,
           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
           "notes": notes}
    if os.path.exists(RESULTS_FILE):
        table = pd.read_csv(RESULTS_FILE)
        table = table[~((table["experiment"] == experiment) & (table["split"] == split))]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    else:
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        table = pd.DataFrame([row])
    table.to_csv(RESULTS_FILE, index=False)
    return table
