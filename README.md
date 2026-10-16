# CHW Clinical QA: a question-answering assistant for community health workers

> Work in progress (Phase 1: data exploration). Results, the live app link and the demo video link will be added later.

A domain-specific QA assistant in English for community health workers (CHWs) in Africa.
A CHW enters a patient scenario and a question. The system returns answers that licensed Nigerian
doctors gave in similar real cases.

## Dataset and license
- **NigBench / AfriHealth-QA** by Intron Health: https://huggingface.co/datasets/intronhealth/nig-bench
- Paper (medRxiv preprint, 2026): https://www.medrxiv.org/content/10.64898/2026.07.05.26356776.full.pdf
- License: **CC BY-NC-SA 4.0** (non-commercial, attribution, ShareAlike). This repository is a
  non-commercial student project and is released under the same license.
- The dataset is gated. **No raw or derived data is committed to this repo.** All data is loaded through code.

## Setup
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Hugging Face token into .env
```
Accept the dataset terms on its Hugging Face page before running anything.

## Project structure
```
notebooks/01_data_exploration.ipynb   # dataset statistics and checks
src/load_data.py                      # loads the text columns of the dataset
results/                              # figures and result tables
```
