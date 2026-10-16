"""
build_app.py - Package everything the web app needs into app_artifacts/.

Run from the project root, after training the final model (notebook 05):
    python -m src.build_app

app_artifacts/ (git-ignored) is uploaded ONCE to a Hugging Face model repository; the Streamlit app
(app/app.py) downloads it from there when it starts. It contains:
  README.md         model card (license, attribution, how it was trained)
  model/            the fine-tuned PubMedBERT model (step 1: topic)
  step2_model/      frozen all-MiniLM-L6-v2 (step 3: answer inside the topic)
  index.npz         step-1 embeddings of all known questions + their intents (numbers, no text)
  paired.npz        step-3 embeddings of the paired questions + their answer ids (no text)
  answers.npz       step-3 embeddings of the answers
  answer_bank.csv   the answers (HASH dataset, CC BY-NC 4.0)
  config.json       thresholds and settings

What gets PUBLISHED when this folder is uploaded: the models, the embeddings and the answer texts.
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
OUT = "app_artifacts"
GITHUB_URL = "https://github.com/Eddydev-ALU/ML_Technique_I_Summative_sti-qa-assistant"

# Model card for the Hugging Face repository (and for model/, replacing the base model's card, which
# sentence-transformers copies unchanged when saving and which would wrongly describe our model and license)
MODEL_CARD = """\
---
base_model: NeuML/pubmedbert-base-embeddings
library_name: sentence-transformers
pipeline_tag: sentence-similarity
license: cc-by-nc-4.0
language: en
---

# STI Q&A Assistant: model and answer files (student project)

Files used by the STI Q&A Assistant web app ({github_url}).

- `model/`: PubMedBERT fine-tuned from [NeuML/pubmedbert-base-embeddings](https://huggingface.co/NeuML/pubmedbert-base-embeddings)
  (Apache-2.0) with MultipleNegativesRankingLoss on (question, same-intent question) and (question, answer) pairs.
  Trained on all 2,502 questions of 109 intents for 4 epochs (batch 32, lr 2e-5, max length 256).
  Evaluation of a model trained the same way on the training split only: test accuracy@1 0.952, MRR 0.968.
- `step2_model/`: [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  (Apache-2.0), unchanged.
- `answer_bank.csv`: the answers, and `*.npz`: embeddings (numbers only, no question text).

**Data:** Dataset Club, HASH (2026). *The HASH Sexually Transmitted Infection Question and Answer Crowdsourced
Dataset* (Version 1). Harvard Dataverse. https://doi.org/10.7910/DVN/5SWM5U. Infectious Diseases Institute
(Uganda), Makerere AI Lab and Sunbird AI. Licensed CC BY-NC 4.0, so these files are for **non-commercial use only**.

**Educational information only, not a diagnosis. Please visit a health facility for testing and treatment.**
"""


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)  # always rebuild from scratch
    os.makedirs(OUT)

    # 1. Step 1: index of ALL known questions, encoded with the final fine-tuned model
    questions, answer_bank = load_prepared()
    encoder = SentenceTransformer(MODEL_DIR, device=get_device())
    retriever = EmbeddingRetriever(encoder, mode="q2q").fit(questions["question"], questions["intent_id"], answer_bank)
    retriever.save_index(os.path.join(OUT, "index.npz"))

    # 2. Step 3: frozen MiniLM embeddings of the paired questions and of the answers
    bank = answer_bank[["answer_id", "intent_id", "category", "simple_answer", "long_answer"]].reset_index(drop=True)
    bank.to_csv(os.path.join(OUT, "answer_bank.csv"), index=False)
    step2 = EmbeddingRetriever(STEP2_MODEL)
    paired = questions[questions["row_answer_id"].notna()]
    np.savez(os.path.join(OUT, "paired.npz"), vectors=step2.encode(paired["question"]),
             answer_ids=np.array(paired["row_answer_id"].astype(str).tolist(), dtype=str))
    np.savez(os.path.join(OUT, "answers.npz"), vectors=step2.encode(bank["simple_answer"]))
    step2.model.save(os.path.join(OUT, "step2_model"))

    # 3. Settings
    with open(THRESHOLD_FILE) as f:
        threshold_info = json.load(f)
    config = {"threshold": threshold_info["threshold"], "low_threshold": threshold_info["low_threshold"],
              "confidence": threshold_info["confidence"],
              "threshold_note": "both tuned in notebook 04 on the evaluation model (trained without val/test/held-out)",
              "step2_model": STEP2_MODEL,
              "n_questions_indexed": len(questions), "n_intents": len(retriever.intents), "n_answers": len(bank)}
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 4. Fine-tuned model + model cards
    shutil.copytree(MODEL_DIR, os.path.join(OUT, "model"))
    for path in [os.path.join(OUT, "README.md"), os.path.join(OUT, "model", "README.md")]:
        with open(path, "w") as f:
            f.write(MODEL_CARD.format(github_url=GITHUB_URL))

    print(json.dumps(config, indent=2))
    for root, _, files in os.walk(OUT):
        for name in sorted(files):
            path = os.path.join(root, name)
            print(f"  {os.path.getsize(path) / 1e6:8.1f} MB  {path}")


if __name__ == "__main__":
    main()
