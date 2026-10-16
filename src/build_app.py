"""
build_app.py - Package everything the web app needs into space_build/ (ready for Hugging Face Spaces).

Run from the project root, after training the final model (notebook 05):
    python -m src.build_app

space_build/ (git-ignored) will contain:
  app.py                       the Gradio app (copied from app/app.py)
  src/                         the code the app imports (inference, models, answer_selection)
  requirements.txt, README.md  for Hugging Face Spaces (README has the Spaces settings header)
  artifacts/model/             the fine-tuned PubMedBERT model (step 1: topic)
  artifacts/step2_model/       frozen all-MiniLM-L6-v2 (step 3: answer inside the topic)
  artifacts/index.npz          step-1 embeddings of all known questions + their intents (numbers, no text)
  artifacts/paired.npz         step-3 embeddings of the paired questions + their answer ids (no text)
  artifacts/answers.npz        step-3 embeddings of the answers
  artifacts/answer_bank.csv    the answers (HASH dataset, CC BY-NC 4.0)
  artifacts/config.json        threshold and settings

What gets PUBLISHED when this folder is uploaded: the model, the embeddings and the answer texts.
The question texts themselves are NOT included.
"""

import json
import os
import shutil

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data_prep import load_prepared
from src.models import EmbeddingRetriever, get_device

MODEL_DIR = "models/final_bi_encoder"
STEP2_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # frozen, picks the answer inside the topic
THRESHOLD_FILE = "results/ood_threshold.json"
OUT = "space_build"

SPACE_REQUIREMENTS = """\
sentence-transformers==6.1.0
transformers==5.20.0
torch==2.14.1
numpy==2.5.4
pandas==3.0.6
"""

SPACE_README = """\
---
title: STI Q&A Assistant
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Answers STI questions with medically reviewed answers
---

# STI Q&A Assistant

A question-answering assistant on sexually transmitted infections for young people and adults in
sub-Saharan Africa. A fine-tuned PubMedBERT bi-encoder matches the user's question to the closest known
question (topic), and a frozen MiniLM model picks the specific answer, written by medical professionals.
It does not generate text. Below a confidence threshold it warns that it is not sure, and below a lower
threshold it refuses.

**Educational information only, not a diagnosis. Please visit a health facility for testing and treatment.**

- Source code, experiments and evaluation: {github_url}
- Data: Dataset Club, HASH (2026). *The HASH Sexually Transmitted Infection Question and Answer Crowdsourced
  Dataset* (Version 1). Harvard Dataverse. https://doi.org/10.7910/DVN/5SWM5U — CC BY-NC 4.0.
  Infectious Diseases Institute (Uganda), Makerere AI Lab and Sunbird AI.
- Models: fine-tuned from [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
  (Apache-2.0); answer selection with [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (Apache-2.0, unchanged).
- License: the answers and the fine-tuned model are derived from a CC BY-NC 4.0 dataset, so this Space
  is for **non-commercial use only**.
- Privacy: questions are not stored or logged.
"""

GITHUB_URL = "https://github.com/Eddydev-ALU/ML_Technique_I_Summative_sti-qa-assistant"

# Replaces the base model's card, which sentence-transformers copies unchanged when saving and which
# would wrongly describe our fine-tuned model (and its license).
MODEL_CARD = """\
---
base_model: NeuML/pubmedbert-base-embeddings
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: cc-by-nc-4.0
language: en
---

# PubMedBERT fine-tuned for STI question matching (student project)

Fine-tuned from [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
(Apache-2.0) with MultipleNegativesRankingLoss on (question, same-intent question) and (question, answer)
pairs from the HASH STI Q&A dataset (Dataset Club, HASH, 2026, https://doi.org/10.7910/DVN/5SWM5U,
CC BY-NC 4.0). Because it is trained on CC BY-NC 4.0 data, it is shared for **non-commercial use only**.

Trained on all 2,502 questions of 109 intents for 4 epochs (batch 32, lr 2e-5, max length 256).
Evaluation (a model trained the same way on the training split only): test accuracy@1 0.952,
MRR 0.968 on 333 held-out phrasings. Details: {github_url}
"""


def main():
    import gradio

    artifacts = os.path.join(OUT, "artifacts")
    if os.path.exists(OUT):
        shutil.rmtree(OUT)  # always rebuild from scratch
    os.makedirs(os.path.join(OUT, "src"))
    os.makedirs(artifacts)

    # 1. Step 1: index of ALL known questions, encoded with the final fine-tuned model
    questions, answer_bank = load_prepared()
    encoder = SentenceTransformer(MODEL_DIR, device=get_device())
    retriever = EmbeddingRetriever(encoder, mode="q2q").fit(questions["question"], questions["intent_id"], answer_bank)
    retriever.save_index(os.path.join(artifacts, "index.npz"))

    # 2. Step 3: frozen MiniLM embeddings of the paired questions and of the answers
    bank = answer_bank[["answer_id", "intent_id", "category", "simple_answer", "long_answer"]].reset_index(drop=True)
    bank.to_csv(os.path.join(artifacts, "answer_bank.csv"), index=False)
    step2 = EmbeddingRetriever(STEP2_MODEL)
    paired = questions[questions["row_answer_id"].notna()]
    np.savez(os.path.join(artifacts, "paired.npz"), vectors=step2.encode(paired["question"]),
             answer_ids=np.array(paired["row_answer_id"].astype(str).tolist(), dtype=str))
    np.savez(os.path.join(artifacts, "answers.npz"), vectors=step2.encode(bank["simple_answer"]))
    step2.model.save(os.path.join(artifacts, "step2_model"))

    # 3. Settings
    with open(THRESHOLD_FILE) as f:
        threshold_info = json.load(f)
    config = {"threshold": threshold_info["threshold"], "low_threshold": threshold_info["low_threshold"],
              "confidence": threshold_info["confidence"],
              "threshold_note": "both tuned in notebook 04 on the evaluation model (trained without val/test/held-out)",
              "step2_model": STEP2_MODEL,
              "n_questions_indexed": len(questions), "n_intents": len(retriever.intents), "n_answers": len(bank)}
    with open(os.path.join(artifacts, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 4. Model, code and Space files
    shutil.copytree(MODEL_DIR, os.path.join(artifacts, "model"))
    with open(os.path.join(artifacts, "model", "README.md"), "w") as f:
        f.write(MODEL_CARD.format(github_url=GITHUB_URL))
    shutil.copy("app/app.py", os.path.join(OUT, "app.py"))
    for name in ["__init__.py", "inference.py", "models.py", "answer_selection.py"]:
        shutil.copy(os.path.join("src", name), os.path.join(OUT, "src", name))
    with open(os.path.join(OUT, "requirements.txt"), "w") as f:
        f.write(SPACE_REQUIREMENTS)
    with open(os.path.join(OUT, "README.md"), "w") as f:
        f.write(SPACE_README.format(gradio_version=gradio.__version__, github_url=GITHUB_URL))

    print(json.dumps(config, indent=2))
    for root, _, files in os.walk(OUT):
        for name in sorted(files):
            path = os.path.join(root, name)
            print(f"  {os.path.getsize(path) / 1e6:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
