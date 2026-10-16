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


# ----------------------------------------------------------------------------
# Out-of-domain ("I don't know") detection
# ----------------------------------------------------------------------------
# The app answers only when the model's confidence is at least a threshold t; otherwise it says
# "I don't know". We treat "should be refused" (out-of-domain) as the POSITIVE class:
#   precision = of the questions we refused, how many really were out-of-domain?
#   recall    = of the out-of-domain questions, how many did we refuse?
def confidence(scores, kind="top1"):
    """Confidence of each prediction from a score matrix.

    top1   : score of the best intent (for embeddings: cosine similarity of the closest question)
    margin : best score minus second-best score (how clearly the winner wins)
    """
    ordered = np.sort(np.asarray(scores), axis=1)[:, ::-1]
    return ordered[:, 0] if kind == "top1" else ordered[:, 0] - ordered[:, 1]


def ood_metrics(conf_in, conf_out, threshold):
    """Detection metrics at one threshold. conf_in: in-domain questions, conf_out: out-of-domain."""
    refused_in = np.asarray(conf_in) < threshold
    refused_out = np.asarray(conf_out) < threshold
    tp, fp, fn = refused_out.sum(), refused_in.sum(), (~refused_out).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "ood_precision": float(precision), "ood_recall": float(recall),
            "ood_f1": float(f1), "in_domain_answered": float(1 - refused_in.mean())}


def choose_threshold(conf_in, conf_out):
    """Threshold with the best OOD F1 on a TUNING set (ties: the lower threshold, i.e. answer more)."""
    candidates = np.unique(np.concatenate([conf_in, conf_out]))
    best = max(candidates, key=lambda t: (ood_metrics(conf_in, conf_out, t)["ood_f1"], -t))
    return float(best)


def threshold_for_coverage(conf_in, coverage=0.95):
    """Alternative rule: the threshold that still answers `coverage` of in-domain questions."""
    return float(np.quantile(conf_in, 1 - coverage))
