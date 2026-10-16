# STI QA Assistant: sexual health question answering for young people in sub-Saharan Africa

> **Live app:** _link added after deployment_ · **Demo video:** _link added later_

A domain-specific question-answering assistant in English on sexual health and sexually
transmitted infections (STIs). A user types a question in everyday language. The system returns an
answer written and reviewed by medical professionals, or says the question is outside its domain.

**Educational information only, not a diagnosis. Please visit a health facility for testing and treatment.**

## Dataset and license
Dataset Club, HASH. (2026). *The HASH Sexually Transmitted Infection Question and Answer Crowdsourced
Dataset* (Version 1) [Data set]. Harvard Dataverse. https://doi.org/10.7910/DVN/5SWM5U

- Produced by the Infectious Diseases Institute (Uganda), Makerere AI Lab and Sunbird AI, under the HASH hub (AI4D).
- Info page: https://www.ai4d.ai/research/the-hash-sexually-transmitted-infection-question-and-answer-crowdsourced-dataset
- Study protocol: Oseku et al., *JMIR Research Protocols* 2025;14:e70005. https://www.researchprotocols.org/2025/1/e70005
- License: **CC BY-NC 4.0** (attribution required, non-commercial use only).
- **No raw or processed data is committed to this repository.** `src/download_data.py` downloads the file.

## Results so far (seen intents, test set: 333 questions)
Settings chosen on the validation set; test reported once. Full table: [results/experiments.md](results/experiments.md).

| Model | Trained? | Test accuracy@1 | Test MRR | Test macro-F1 |
|---|---|---|---|---|
| Majority class | – | 0.042 | 0.128 | 0.001 |
| BM25 (question → question) | – | 0.360 | 0.540 | 0.255 |
| TF-IDF (char 2–5) + logistic regression | yes | 0.823 | 0.867 | 0.610 |
| MiniLM embeddings, frozen | no | 0.844 | 0.896 | 0.688 |
| PubMedBERT embeddings, frozen | no | 0.712 | 0.798 | 0.526 |
| DistilBERT classifier, fine-tuned | yes | 0.940 | 0.952 | 0.764 |
| MiniLM bi-encoder, fine-tuned (MNRL) | yes | 0.940 | 0.959 | 0.808 |
| **PubMedBERT bi-encoder, fine-tuned (MNRL)** | yes | **0.952** | **0.968** | **0.835** |

Seed study (3 seeds each, test acc@1): PubMedBERT 0.954 ± 0.007, MiniLM 0.941 ± 0.002, DistilBERT 0.940 ± 0.003
([results/seed_study.md](results/seed_study.md)).

## Evaluation of the final model (fine-tuned PubMedBERT)
**"I don't know" (out-of-domain) detection.** Threshold on the top-1 cosine similarity, chosen on a tuning
half (val + 5 held-out intents + 30 off-topic questions) and reported on a separate test half
(test + 6 other held-out intents + 30 other off-topic questions). Full table: [results/ood_results.csv](results/ood_results.csv).

| Test half | n | Refused |
|---|---|---|
| Unseen STI topics (held-out intents) | 160 | 98.1% |
| Off-topic: unrelated / health-not-STI / near-domain | 10 / 10 / 10 | 100% / 100% / 100% |
| Real in-domain questions (should be answered) | 333 | 6.9% |

With the threshold, the app answers 93% of real test questions, with 98.1% of those answers correct.
The off-topic questions were written by an AI assistant for testing only (see [evaluation/README.md](evaluation/README.md)).

**"Not sure" zone.** Real users' wording scores lower than the dataset's uniform paraphrases, so the app has a
second, lower threshold (0.728, fixed on the tuning half). Test half: in-domain questions refused outright drop
from 6.9% to 1.2% (5.7% get a "Did you mean …?" answer instead), while all 30 off-topic test questions are
still refused. Table: [results/ood_three_zones.csv](results/ood_three_zones.csv).

**Choosing one answer inside multi-answer intents** (99 val/test questions, 7.6 answers per intent on average):

| Method | Accuracy |
|---|---|
| Random (expected) / always the first answer | 0.192 / 0.202 |
| Answer-text matching, fine-tuned PubMedBERT (the step-1 model) | 0.465 |
| Answer-text matching, frozen PubMedBERT | 0.505 |
| **Answer-text matching, frozen MiniLM** (used in the app) | **0.556** |

Fine-tuning pulls all questions of an intent together, which helps find the intent but hides the differences
*inside* an intent; a frozen model keeps them. The app therefore uses two encoders.

## How the deployed system works
1. **Topic (step 1):** a fine-tuned PubMedBERT bi-encoder (`models/final_bi_encoder`, trained on all 2,502
   questions of 109 intents in notebook 05) finds the most similar known question.
2. **Decision (step 2):** confidence ≥ 0.854 → answer; 0.728–0.854 → "Did you mean a question about …?";
   below 0.728 → "I can only answer questions about STIs …".
3. **Answer (step 3):** a frozen MiniLM picks the answer inside the topic (the dataset's own paired answer if the
   question is near-identical to a paired question, otherwise the most similar answer text).
4. The app shows the answer (simple or detailed), the matched topic, the confidence and the top-3 topics,
   with a permanent disclaimer. **Questions are not stored or logged.** No text is generated.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data     # saves data/raw/hash_sti_qa.xlsx (git-ignored), checks the MD5
python -m src.data_prep         # cleans data, builds the answer bank, recreates the splits
```
The split assignment (question IDs only, no text) is committed in `splits/`, so everyone gets identical splits.

**Reproduce the experiments:** run notebooks `01` → `05` in order (`03`, `03b` and `05` need a GPU; they are
Colab-ready: open them from GitHub in Colab and choose a T4 GPU).

**Run the web app locally** (needs `models/final_bi_encoder` from notebook 05):
```bash
python -m src.build_app     # packages model, embeddings and answers into space_build/ (git-ignored)
python app/app.py           # opens the Gradio app at http://127.0.0.1:7860
```

**Deploy to Hugging Face Spaces:** `space_build/` is a complete Space (app, code, requirements, README with the
Spaces header, artifacts). Upload it with `hf upload <user>/<space> space_build . --repo-type space`.

## Project structure
```
src/download_data.py                  # downloads the dataset from Harvard Dataverse
src/load_raw.py                       # reads the 8 Excel sheets into one table
src/data_prep.py                      # cleaning, answer bank, train/val/test + held-out-intent splits
src/baselines.py                      # Majority, BM25 and TF-IDF + logistic regression baselines
src/evaluate.py                       # shared metrics (accuracy@1, recall@k, MRR, macro-F1)
src/models.py                         # embedding retriever (q2q / q2a) and Transformer classifier
src/train.py                          # fine-tuning: MNRL bi-encoder and DistilBERT classifier
src/answer_selection.py               # choose one answer inside the predicted intent
src/inference.py                      # answer(question, detail): the deployed question-answering logic
src/build_app.py                      # packages everything the app needs into space_build/
app/app.py                            # Gradio web interface
splits/                               # question_id -> split, held-out intent list (no data text)
notebooks/01_data_exploration.ipynb   # dataset statistics and checks
notebooks/02_baselines.ipynb          # baselines: tuning on val, results on test, errors
notebooks/03_finetune.ipynb           # main models and experiments (run on Colab T4 GPU)
notebooks/03b_seed_study.ipynb        # 3 seeds per fine-tuned model (run on Colab T4 GPU)
notebooks/04_error_analysis.ipynb     # final model: errors, out-of-domain thresholds, answer selection
notebooks/05_final_model.ipynb        # retrain the deployed model on all intents (Colab T4 GPU)
evaluation/                           # 60 out-of-domain test questions (AI-written, disclosed)
results/                              # figures and result tables
```
