"""
load_data.py - Load the TEXT columns of the NigBench (AfriHealth-QA) dataset.

Dataset : https://huggingface.co/datasets/intronhealth/nig-bench
License : CC BY-NC-SA 4.0 (Intron Health). Non-commercial, attribute, ShareAlike.

The dataset is gated: accept its terms on the Hugging Face website, then put your
token in a `.env` file as HF_TOKEN=... (see .env.example). Nothing is hardcoded.

We never save the data inside the repo. The `datasets` library caches the download
in ~/.cache/huggingface, which is outside the project folder.
"""

import os

import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

DATASET_NAME = "intronhealth/nig-bench"

# Only the text columns we need. The dataset also has audio, image and video
# columns that make up most of its size; passing `columns=` means they are
# never decoded into memory.
TEXT_COLUMNS = [
    "question_id",   # id of the question (one question can have several answers)
    "answer_id",     # id of a single doctor answer
    "category",      # CHW curriculum topic
    "project_name",  # data collection project the row came from
    "prompt",        # (not described on the card) inspected in notebook 01
    "scenario",      # patient scenario written for the question
    "question",      # the question asked about the scenario
    "answer",        # free-text answer written by a licensed Nigerian doctor
    "difficulty",    # Easy / Medium / Hard
    "language",      # language label of the row
    "src_lang",      # translation fields: source language ...
    "src_text",      # ... source text ...
    "target_lang",   # ... target language ...
    "target_text",   # ... and target (translated) text
]


def get_hf_token():
    """Read the Hugging Face token from the .env file (or the environment)."""
    load_dotenv()  # loads variables from .env into os.environ, if the file exists
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN not found. Copy .env.example to .env and add your token "
            "(and accept the dataset terms on the Hugging Face website)."
        )
    return token


def load_text_data():
    """Download (first time only) and return the text columns as a pandas DataFrame."""
    dataset = load_dataset(
        DATASET_NAME,
        split="train",          # the dataset ships as a single 'train' split
        columns=TEXT_COLUMNS,   # skip audio/image/video columns
        token=get_hf_token(),
    )
    return dataset.to_pandas()


if __name__ == "__main__":
    # Quick check: python -m src.load_data
    df = load_text_data()
    print(df.shape)
    print(df.head())
