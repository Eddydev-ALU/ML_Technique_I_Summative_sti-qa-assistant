"""
Cleans the HASH data, builds the answer bank and creates the splits.

Usage: python -m src.data_prep   (after src.download_data)

Steps: fix intent labels -> answer bank -> questions table -> remove duplicates ->
group near-identical questions -> splits. The splits are created once and then reused from
splits/splits.csv (pass --resplit to make new ones). Cleaned text goes to data/processed/
(git-ignored), only the split ids and a summary with counts are committed.
"""

import json
import os
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.load_raw import load_raw_sheets

SEED = 42
HELD_OUT_FRACTION = 0.10
MIN_QUESTIONS_TO_HOLD_OUT = 5  # held-out intents need enough questions to test on
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_GROUPS_TO_SPLIT = 3        # need at least one group each for train, val and test
NEAR_DUP_THRESHOLD = 0.90      # char n-gram cosine similarity

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = "splits"
RESULTS_DIR = "results"


# --- text cleaning ---
def clean_text(text):
    """Normalise unicode and whitespace but keep line breaks (answers have bullet points)."""
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text))  # e.g. non-breaking spaces
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def clean_question(text):
    return " ".join(clean_text(text).split())


def normalise_for_matching(text):
    """Lowercase without punctuation, only used to find duplicates."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


# --- intent labels ---
def assign_intents(raw):
    """Final intent per row. Blocks labelled just Yes/No, and blocks with no answer at all,
    are merged into the block above them."""
    raw = raw.copy()
    raw["intent_label"] = raw["intent_block"].str.strip()

    has_answer = raw["simple_answer"].notna() | raw["long_answer"].notna()
    block_has_answer = has_answer.groupby([raw["category"], raw["intent_block"]]).transform("any")

    final_intent = []
    merged_blocks = set()
    previous = None
    for i, row in raw.iterrows():
        if i == 0 or row["category"] != raw.at[i - 1, "category"]:
            previous = None  # new sheet
        starts_block = pd.notna(row["intent"])
        if starts_block:
            is_yes_no = row["intent_label"].lower() in ("yes", "no")
            should_merge = (is_yes_no or not block_has_answer[i]) and previous is not None
            current = previous if should_merge else row["intent_label"]
            if should_merge:
                merged_blocks.add((row["category"], row["intent_label"], previous))
        final_intent.append(current)
        previous = current

    raw["intent_label"] = final_intent
    # same intent names appear in several sheets, so prefix the category
    raw["intent_id"] = raw["category"] + "::" + raw["intent_label"]
    return raw, sorted(merged_blocks)


# --- answer bank ---
def build_answer_bank(raw):
    """One entry per answer, with its intent and the question on the same row (if any)."""
    entries = []
    for intent_id, block in raw.groupby("intent_id", sort=False):
        for _, row in block.iterrows():
            simple, long = clean_text(row["simple_answer"]), clean_text(row["long_answer"])
            if not simple and not long:
                continue
            row_question = clean_question(row["q_crowd"]) or clean_question(row["q_other"])

            if not row_question and entries and entries[-1]["intent_id"] == intent_id:
                # no question on this row: it's the rest of the previous answer (bullet points)
                prev = entries[-1]
                prev["simple_answer"] = "\n".join(t for t in [prev["simple_answer"], simple] if t)
                prev["long_answer"] = "\n".join(t for t in [prev["long_answer"], long] if t)
                continue

            entries.append({
                "answer_id": f"{row['category']}|r{row['excel_row']}",
                "intent_id": intent_id,
                "category": row["category"],
                "excel_row": row["excel_row"],
                "anchor_question": row_question,
                "simple_answer": simple,
                "long_answer": long,
            })

    bank = pd.DataFrame(entries)
    # if only one version exists, use it for both
    bank["simple_missing"] = bank["simple_answer"] == ""
    bank["long_missing"] = bank["long_answer"] == ""
    bank.loc[bank["simple_missing"], "simple_answer"] = bank["long_answer"]
    bank.loc[bank["long_missing"], "long_answer"] = bank["simple_answer"]
    bank["is_default"] = ~bank["intent_id"].duplicated()
    bank["answers_in_intent"] = bank.groupby("intent_id")["answer_id"].transform("count")
    return bank


# --- questions ---
def build_questions(raw, bank):
    """One row per question, with its intent and the answer on its row (if any)."""
    answer_on_row = dict(zip(zip(bank["category"], bank["excel_row"]), bank["answer_id"]))

    rows = []
    for _, row in raw.iterrows():
        for col, source in [("q_crowd", "crowdsourced"), ("q_other", "another_way")]:
            if pd.isna(row[col]):
                continue
            # a few cells contain several questions on separate lines
            for line_no, line in enumerate(str(row[col]).split("\n")):
                question = clean_question(line)
                if not question:
                    continue
                rows.append({
                    "question_id": f"{row['category']}|r{row['excel_row']}|{source}|{line_no}",
                    "category": row["category"],
                    "intent_id": row["intent_id"],
                    "question": question,
                    "source": source,
                    "excel_row": row["excel_row"],
                    "row_answer_id": answer_on_row.get((row["category"], row["excel_row"])),
                })
    return pd.DataFrame(rows)


def remove_duplicates(questions):
    """Keep one copy of repeated questions. Questions that appear under different intents are
    dropped completely since there's no way to know which label is right."""
    questions = questions.copy()
    questions["norm"] = questions["question"].map(normalise_for_matching)

    intents_per_text = questions.groupby("norm")["intent_id"].transform("nunique")
    conflicting = questions[intents_per_text > 1]
    questions = questions[intents_per_text == 1]

    before = len(questions)
    questions = questions.drop_duplicates(subset=["norm", "intent_id"], keep="first")
    stats = {
        "conflicting_texts_dropped": int(conflicting["norm"].nunique()),
        "conflicting_rows_dropped": int(len(conflicting)),
        "exact_duplicate_rows_dropped": int(before - len(questions)),
    }
    return questions.reset_index(drop=True), stats


# --- near-duplicate groups ---
def find_root(parent, i):
    """Union-find."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def add_near_duplicate_groups(questions):
    """Near-identical questions get the same group_id, so a whole group goes into one split
    (otherwise test would contain copies of training questions)."""
    parent = list(range(len(questions)))

    # only within a category: "...from Herpes?" and "...from HIV?" are different questions
    for _, cat_rows in questions.groupby("category"):
        idx = cat_rows.index.to_numpy()
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(cat_rows["norm"])
        sim = cosine_similarity(vectors)
        a, b = np.where(np.triu(sim, k=1) >= NEAR_DUP_THRESHOLD)
        for i, j in zip(idx[a], idx[b]):
            parent[find_root(parent, i)] = find_root(parent, j)

    roots = [find_root(parent, i) for i in range(len(questions))]
    questions = questions.copy()
    questions["group_id"] = pd.factorize(pd.Series(roots))[0]
    questions["group_intent"] = questions.groupby("group_id")["intent_id"].transform(lambda s: s.mode().iloc[0])
    return questions


# --- splits ---
def make_splits(questions):
    """Split per question (train/val/test/held_out) and the list of held-out intents."""
    rng = np.random.default_rng(SEED)

    # hold out ~10% of the intents completely
    q_per_intent = questions.groupby("intent_id").size()
    candidates = sorted(q_per_intent[q_per_intent >= MIN_QUESTIONS_TO_HOLD_OUT].index)
    n_hold = round(HELD_OUT_FRACTION * len(q_per_intent))
    held_out = sorted(rng.choice(candidates, size=n_hold, replace=False))

    # 70/15/15 of the groups of every other intent
    group_split = {}
    groups = questions.drop_duplicates("group_id")[["group_id", "group_intent"]]
    for intent_id, intent_groups in groups.groupby("group_intent", sort=True):
        ids = intent_groups["group_id"].to_numpy()
        if intent_id in held_out:
            group_split.update({g: "held_out" for g in ids})
            continue
        ids = rng.permutation(ids)
        if len(ids) < MIN_GROUPS_TO_SPLIT:
            group_split.update({g: "train" for g in ids})
            continue
        n_val = max(1, round(VAL_FRACTION * len(ids)))
        n_test = max(1, round(TEST_FRACTION * len(ids)))
        group_split.update({g: "test" for g in ids[:n_test]})
        group_split.update({g: "val" for g in ids[n_test:n_test + n_val]})
        group_split.update({g: "train" for g in ids[n_test + n_val:]})

    split = questions["group_id"].map(group_split)
    return split, held_out


def load_prepared():
    """Questions with their split, and the answer bank (used by the notebooks and training)."""
    questions = pd.read_csv(os.path.join(PROCESSED_DIR, "questions.csv"))
    splits = pd.read_csv(os.path.join(SPLITS_DIR, "splits.csv"))
    bank = pd.read_csv(os.path.join(PROCESSED_DIR, "answer_bank.csv"), keep_default_na=False)
    if set(questions["question_id"]) != set(splits["question_id"]):
        raise RuntimeError("Processed questions and splits/splits.csv do not match. Re-run: python -m src.data_prep")
    questions = questions.merge(splits, on="question_id", how="inner", validate="one_to_one")
    return questions, bank


def main(resplit=False):
    raw = load_raw_sheets()
    raw, merged_blocks = assign_intents(raw)
    bank = build_answer_bank(raw)
    questions = build_questions(raw, bank)
    n_questions_raw = len(questions)
    questions, dup_stats = remove_duplicates(questions)
    questions = add_near_duplicate_groups(questions)

    splits_file = os.path.join(SPLITS_DIR, "splits.csv")
    held_out_file = os.path.join(SPLITS_DIR, "held_out_intents.txt")
    if os.path.exists(splits_file) and not resplit:
        # reuse the committed splits so Colab and my laptop evaluate on the same questions
        frozen = pd.read_csv(splits_file)
        if set(frozen["question_id"]) != set(questions["question_id"]):
            raise RuntimeError("The cleaned questions no longer match splits/splits.csv. "
                               "Check the data file, or run with --resplit to create new splits.")
        questions["split"] = questions["question_id"].map(dict(zip(frozen["question_id"], frozen["split"])))
        with open(held_out_file) as f:
            held_out = [line.strip() for line in f if line.strip()]
        print("Using the committed splits in splits/ (run with --resplit to create new ones)")
    else:
        questions["split"], held_out = make_splits(questions)
        os.makedirs(SPLITS_DIR, exist_ok=True)
        questions[["question_id", "split"]].to_csv(splits_file, index=False)
        with open(held_out_file, "w") as f:
            f.write("\n".join(held_out) + "\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    keep = ["question_id", "category", "intent_id", "question", "source", "row_answer_id", "group_id"]
    questions[keep].to_csv(os.path.join(PROCESSED_DIR, "questions.csv"), index=False)
    bank.to_csv(os.path.join(PROCESSED_DIR, "answer_bank.csv"), index=False)

    seen = questions[questions["split"] != "held_out"]
    q_per_intent = seen.groupby("intent_id").size()
    group_sizes = questions.groupby("group_id").size()
    summary = {
        "seed": SEED,
        "merged_blocks": [f"{c}::{label} -> {target}" for c, label, target in merged_blocks],
        "intents_total": int(questions["intent_id"].nunique()),
        "intents_held_out": len(held_out),
        "intents_seen": int(seen["intent_id"].nunique()),
        "intents_train_only": int((seen.groupby("intent_id")["split"].nunique() == 1).sum()),
        "questions_after_splitting_cells": n_questions_raw,
        **dup_stats,
        "questions_final": len(questions),
        "near_duplicate_groups": int(len(group_sizes)),
        "questions_in_multi_member_groups": int(group_sizes[group_sizes > 1].sum()),
        "largest_group_size": int(group_sizes.max()),
        "groups_with_mixed_intents": int((questions.groupby("group_id")["intent_id"].nunique() > 1).sum()),
        "split_counts": questions["split"].value_counts().to_dict(),
        "seen_questions_per_intent": {k: float(v) for k, v in q_per_intent.describe()[["min", "50%", "max"]].items()},
        "answers_total": len(bank),
        "intents_with_one_answer": int((bank.groupby("intent_id").size() == 1).sum()),
        "intents_with_several_answers": int((bank.groupby("intent_id").size() > 1).sum()),
        "answers_simple_missing_filled": int(bank["simple_missing"].sum()),
        "answers_long_missing_filled": int(bank["long_missing"].sum()),
        "questions_with_row_answer": int(questions["row_answer_id"].notna().sum()),
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("\nHeld-out intents:", *held_out, sep="\n  ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--resplit", action="store_true", help="create new splits instead of reusing splits/")
    main(resplit=parser.parse_args().resplit)
