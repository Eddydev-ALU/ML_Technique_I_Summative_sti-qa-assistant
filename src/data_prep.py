"""
data_prep.py - Clean the HASH STI Q&A data, build the answer bank, and create the splits.

Run from the project root (after python -m src.download_data):
    python -m src.data_prep

What it does, step by step:
  1. Read the 8 Excel sheets (src/load_raw.py).
  2. Fix intent labels: strip spaces; merge "Yes"/"No" sub-blocks and blocks without
     any answer into the intent block just above them.
  3. Build the ANSWER BANK: one entry per answer. An answer row without a question is a
     continuation (bullet points spilling onto the next row) and is appended to the
     answer above it.
  4. Build the QUESTIONS table: one row per question, cleaned. Cells holding several
     questions on separate lines are split. Duplicates are removed: a repeated question
     under the same intent is kept once; a question that appears under DIFFERENT intents
     is dropped, because we cannot know which label is right.
  5. Group near-identical questions (e.g. only punctuation differs) so that a group always
     lands in ONE split. Otherwise the test set would contain copies of training questions.
  6. Splits (fixed seed). Created once, then FROZEN: later runs reuse splits/splits.csv
     (use --resplit to create new ones):
       held_out : ~10% of intents removed entirely (tests "I don't know" on unseen topics)
       train / val / test : 70/15/15 of the groups of every remaining intent
     Intents with fewer than 3 groups go entirely to train.

Outputs:
  data/processed/questions.csv    question text + labels      (git-ignored: contains data)
  data/processed/answer_bank.csv  answers per intent          (git-ignored: contains data)
  splits/splits.csv               question_id -> split        (committed: IDs only, no text)
  splits/held_out_intents.txt     the held-out intents        (committed)
  results/data_summary.json       counts only                 (committed)
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
HELD_OUT_FRACTION = 0.10      # share of intents held out entirely
MIN_QUESTIONS_TO_HOLD_OUT = 5  # only hold out intents with enough questions to test on
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
MIN_GROUPS_TO_SPLIT = 3       # an intent needs >= 3 groups to give one each to train/val/test
NEAR_DUP_THRESHOLD = 0.90     # cosine similarity (character n-grams) for "near-identical"

PROCESSED_DIR = os.path.join("data", "processed")
SPLITS_DIR = "splits"
RESULTS_DIR = "results"


# ----------------------------------------------------------------------------
# Text cleaning helpers
# ----------------------------------------------------------------------------
def clean_text(text):
    """Fix encoding oddities and spacing, keeping line breaks (used for answers)."""
    if pd.isna(text):
        return ""
    # NFKC turns odd characters (e.g. non-breaking spaces) into their normal form
    text = unicodedata.normalize("NFKC", str(text))
    lines = [" ".join(line.split()) for line in text.split("\n")]  # collapse spaces/tabs per line
    return "\n".join(line for line in lines if line)                 # drop empty lines


def clean_question(text):
    """Clean a single question: one line, single spaces."""
    return " ".join(clean_text(text).split())


def normalise_for_matching(text):
    """Lower-case, no punctuation. Only used to detect duplicates, never shown or trained on."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


# ----------------------------------------------------------------------------
# Step 2: intent labels
# ----------------------------------------------------------------------------
def assign_intents(raw):
    """Give every row its final intent, after merging Yes/No and answer-less blocks upward."""
    raw = raw.copy()
    raw["intent_label"] = raw["intent_block"].str.strip()

    # Which blocks have at least one answer?
    has_answer = raw["simple_answer"].notna() | raw["long_answer"].notna()
    block_has_answer = has_answer.groupby([raw["category"], raw["intent_block"]]).transform("any")

    # Walk down each sheet; a block that should be merged takes the previous block's intent
    final_intent = []
    merged_blocks = set()
    previous = None
    for i, row in raw.iterrows():
        if i == 0 or row["category"] != raw.at[i - 1, "category"]:
            previous = None  # new sheet: nothing to merge into
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
    # Intent names repeat across categories (e.g. 'definition_infections'), so prefix the category
    raw["intent_id"] = raw["category"] + "::" + raw["intent_label"]
    return raw, sorted(merged_blocks)


# ----------------------------------------------------------------------------
# Step 3: answer bank
# ----------------------------------------------------------------------------
def build_answer_bank(raw):
    """One entry per answer, with its intent and the question written on the same row (if any)."""
    entries = []
    for intent_id, block in raw.groupby("intent_id", sort=False):
        for _, row in block.iterrows():
            simple, long = clean_text(row["simple_answer"]), clean_text(row["long_answer"])
            if not simple and not long:
                continue
            row_question = clean_question(row["q_crowd"]) or clean_question(row["q_other"])

            if not row_question and entries and entries[-1]["intent_id"] == intent_id:
                # Continuation row: append to the previous answer of this intent
                prev = entries[-1]
                prev["simple_answer"] = "\n".join(t for t in [prev["simple_answer"], simple] if t)
                prev["long_answer"] = "\n".join(t for t in [prev["long_answer"], long] if t)
                continue

            entries.append({
                "answer_id": f"{row['category']}|r{row['excel_row']}",
                "intent_id": intent_id,
                "category": row["category"],
                "excel_row": row["excel_row"],
                "anchor_question": row_question,  # question on the same row (used in step 2 of the model)
                "simple_answer": simple,
                "long_answer": long,
            })

    bank = pd.DataFrame(entries)
    # If one of the two answers is missing, fall back to the other one (and record that we did)
    bank["simple_missing"] = bank["simple_answer"] == ""
    bank["long_missing"] = bank["long_answer"] == ""
    bank.loc[bank["simple_missing"], "simple_answer"] = bank["long_answer"]
    bank.loc[bank["long_missing"], "long_answer"] = bank["simple_answer"]
    # The first answer of each intent is its default answer
    bank["is_default"] = ~bank["intent_id"].duplicated()
    bank["answers_in_intent"] = bank.groupby("intent_id")["answer_id"].transform("count")
    return bank


# ----------------------------------------------------------------------------
# Step 4: questions
# ----------------------------------------------------------------------------
def build_questions(raw, bank):
    """One row per cleaned question, with its intent and (if on an answer row) that answer."""
    answer_on_row = dict(zip(zip(bank["category"], bank["excel_row"]), bank["answer_id"]))

    rows = []
    for _, row in raw.iterrows():
        for col, source in [("q_crowd", "crowdsourced"), ("q_other", "another_way")]:
            if pd.isna(row[col]):
                continue
            # A few cells hold several questions on separate lines: split them
            for line_no, line in enumerate(str(row[col]).split("\n")):
                question = clean_question(line)
                if not question:
                    continue
                rows.append({
                    # Stable ID built from the position in the Excel file (never from the text)
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
    """Drop repeated questions; drop questions that carry conflicting intent labels."""
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


# ----------------------------------------------------------------------------
# Step 5: near-duplicate groups
# ----------------------------------------------------------------------------
def find_root(parent, i):
    """Union-find helper: follow parent links to the group's representative."""
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def add_near_duplicate_groups(questions):
    """Give each question a group_id; near-identical questions in a category share one."""
    parent = list(range(len(questions)))

    # Compare only within the same category. Across categories the questions differ in the
    # disease name ("...from Herpes?" vs "...from HIV?"), which are different questions.
    for _, cat_rows in questions.groupby("category"):
        idx = cat_rows.index.to_numpy()
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit_transform(cat_rows["norm"])
        sim = cosine_similarity(vectors)
        a, b = np.where(np.triu(sim, k=1) >= NEAR_DUP_THRESHOLD)
        for i, j in zip(idx[a], idx[b]):
            parent[find_root(parent, i)] = find_root(parent, j)  # join the two groups

    roots = [find_root(parent, i) for i in range(len(questions))]
    questions = questions.copy()
    questions["group_id"] = pd.factorize(pd.Series(roots))[0]
    # A group's intent = the most common intent among its members (used only to place the group)
    questions["group_intent"] = questions.groupby("group_id")["intent_id"].transform(lambda s: s.mode().iloc[0])
    return questions


# ----------------------------------------------------------------------------
# Step 6: splits
# ----------------------------------------------------------------------------
def make_splits(questions):
    """Return a Series question_id -> train/val/test/held_out, and the held-out intents."""
    rng = np.random.default_rng(SEED)

    # (a) Hold out ~10% of intents (only those with enough questions to test on)
    q_per_intent = questions.groupby("intent_id").size()
    candidates = sorted(q_per_intent[q_per_intent >= MIN_QUESTIONS_TO_HOLD_OUT].index)
    n_hold = round(HELD_OUT_FRACTION * len(q_per_intent))
    held_out = sorted(rng.choice(candidates, size=n_hold, replace=False))

    # (b) Split the groups of every other intent 70/15/15
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


# ----------------------------------------------------------------------------
# Helper for later phases
# ----------------------------------------------------------------------------
def load_prepared():
    """Load questions (with their split) and the answer bank. Used by notebooks and training."""
    questions = pd.read_csv(os.path.join(PROCESSED_DIR, "questions.csv"))
    splits = pd.read_csv(os.path.join(SPLITS_DIR, "splits.csv"))
    bank = pd.read_csv(os.path.join(PROCESSED_DIR, "answer_bank.csv"), keep_default_na=False)
    if set(questions["question_id"]) != set(splits["question_id"]):
        raise RuntimeError("Processed questions and splits/splits.csv do not match. Re-run: python -m src.data_prep")
    questions = questions.merge(splits, on="question_id", how="inner", validate="one_to_one")
    return questions, bank


# ----------------------------------------------------------------------------
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
        # The splits are FROZEN once created and committed: reuse them, so every machine
        # (laptop, Colab, a marker's computer) evaluates on exactly the same test questions,
        # even if library versions differ slightly.
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

    # Save cleaned data (git-ignored: contains the dataset text)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    keep = ["question_id", "category", "intent_id", "question", "source", "row_answer_id", "group_id"]
    questions[keep].to_csv(os.path.join(PROCESSED_DIR, "questions.csv"), index=False)
    bank.to_csv(os.path.join(PROCESSED_DIR, "answer_bank.csv"), index=False)

    # Summary (numbers only)
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
