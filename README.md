# STI QA Assistant: sexual health question answering for young people in sub-Saharan Africa

> Work in progress (Phase 1: data exploration). Results, the live app link and the demo video link will be added later.

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

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.download_data     # saves data/raw/hash_sti_qa.xlsx (git-ignored), checks the MD5
python -m src.data_prep         # cleans data, builds the answer bank, recreates the splits
```
The split assignment (question IDs only, no text) is committed in `splits/`, so everyone gets identical splits.

## Project structure
```
src/download_data.py                  # downloads the dataset from Harvard Dataverse
src/load_raw.py                       # reads the 8 Excel sheets into one table
src/data_prep.py                      # cleaning, answer bank, train/val/test + held-out-intent splits
src/baselines.py                      # Majority, BM25 and TF-IDF + logistic regression baselines
src/evaluate.py                       # shared metrics (accuracy@1, recall@k, MRR, macro-F1)
src/models.py                         # embedding retriever (q2q / q2a) and Transformer classifier
src/train.py                          # fine-tuning: MNRL bi-encoder and DistilBERT classifier
splits/                               # question_id -> split, held-out intent list (no data text)
notebooks/01_data_exploration.ipynb   # dataset statistics and checks
notebooks/02_baselines.ipynb          # baselines: tuning on val, results on test, errors
notebooks/03_finetune.ipynb           # main models and experiments (run on Colab T4 GPU)
results/                              # figures and result tables
```
