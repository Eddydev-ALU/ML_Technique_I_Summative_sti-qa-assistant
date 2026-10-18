"""
Metrics used for every model, computed from a score matrix (questions x intents):
accuracy@1, recall@3/5, MRR and macro-F1, plus the out-of-domain threshold functions.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RESULTS_FILE = os.path.join("results", "experiments.csv")


def true_ranks(scores, true_idx):
    """Rank of the correct intent (1 = best). Ties count against the model, otherwise
    BM25 giving every intent 0 would look perfect."""
    true_scores = scores[np.arange(len(true_idx)), true_idx]
    return (scores >= true_scores[:, None]).sum(axis=1)


def ranking_metrics(scores, true_idx, ks=(3, 5)):
    """accuracy@1, recall@k, MRR and macro-F1 for one score matrix."""
    scores = np.asarray(scores, dtype=float)
    true_idx = np.asarray(true_idx)
    ranks = true_ranks(scores, true_idx)
    top1 = scores.argmax(axis=1)

    metrics = {"n": len(true_idx), "accuracy@1": float(np.mean(ranks == 1))}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks <= k))
    metrics["mrr"] = float(np.mean(1.0 / ranks))
    # macro-F1 over the intents present in this split, so small intents count as much as big ones
    metrics["macro_f1"] = float(f1_score(true_idx, top1, labels=np.unique(true_idx),
                                         average="macro", zero_division=0))
    return metrics


def per_category_accuracy(scores, true_idx, categories):
    """Accuracy@1 per category."""
    correct = np.asarray(scores).argmax(axis=1) == np.asarray(true_idx)
    table = pd.DataFrame({"category": categories, "correct": correct})
    return table.groupby("category")["correct"].agg(n="size", accuracy="mean").round(3)


def save_result(experiment, split, metrics, notes=""):
    """Add or replace a row in results/experiments.csv."""
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


# --- out-of-domain detection ---
# A question is refused when its confidence is below the threshold. "Refuse" is the positive
# class, so precision = refused questions that really were out-of-domain, recall = out-of-domain
# questions that got refused.
def confidence(scores, kind="top1"):
    """top1 = best intent score, margin = best minus second best."""
    ordered = np.sort(np.asarray(scores), axis=1)[:, ::-1]
    return ordered[:, 0] if kind == "top1" else ordered[:, 0] - ordered[:, 1]


def ood_metrics(conf_in, conf_out, threshold):
    """Precision/recall/F1 of refusing, at one threshold."""
    refused_in = np.asarray(conf_in) < threshold
    refused_out = np.asarray(conf_out) < threshold
    tp, fp, fn = refused_out.sum(), refused_in.sum(), (~refused_out).sum()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "ood_precision": float(precision), "ood_recall": float(recall),
            "ood_f1": float(f1), "in_domain_answered": float(1 - refused_in.mean())}


def choose_threshold(conf_in, conf_out):
    """Threshold with the best OOD F1 (on the tuning half). On ties take the lower one."""
    candidates = np.unique(np.concatenate([conf_in, conf_out]))
    best = max(candidates, key=lambda t: (ood_metrics(conf_in, conf_out, t)["ood_f1"], -t))
    return float(best)


def threshold_for_coverage(conf_in, coverage=0.95):
    """Alternative: threshold that still answers `coverage` of in-domain questions."""
    return float(np.quantile(conf_in, 1 - coverage))
