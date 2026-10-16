"""
train.py - Fine-tuning loops for the two neural models.

1. train_bi_encoder: fine-tune a sentence-embedding model with MultipleNegativesRankingLoss (MNRL).

   Training data = pairs (anchor, positive) that should be close in meaning:
     q2q pairs: (question, another training question of the SAME intent)
     q2a pairs: (question, an answer of its intent)
   MNRL, for a batch of N pairs:
     - embed the N anchors and the N positives
     - build the N x N cosine-similarity matrix (anchor i vs positive j)
     - for anchor i, the correct "class" is positive i; the other N-1 positives in the batch act
       as NEGATIVES ("wrong answers"). Loss = cross-entropy over each row.
   So the model learns: pull a question towards same-intent texts, push it away from others.
   Important: if two pairs in a batch had the same intent, a "negative" would actually be correct.
   We therefore build batches where every pair has a DIFFERENT intent.

2. train_classifier: fine-tune DistilBERT with a classification head (one output per intent),
   using ordinary cross-entropy loss.

Both loops: AdamW optimiser, linear learning-rate warm-up then decay, evaluation on val after
every epoch, EARLY STOPPING (stop when val accuracy has not improved for `patience` epochs, and
keep the best epoch's weights). Train loss, val loss and val accuracy are logged per epoch so we
can see overfitting (train loss keeps falling while val stops improving).
"""

import copy
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.evaluate import ranking_metrics
from src.models import EmbeddingRetriever, TransformerClassifier, get_device

MNRL_SCALE = 20.0  # multiplies cosine similarities before softmax (same default as sentence-transformers)


def free_memory():
    """Release GPU memory held by models that are no longer used."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def set_seed(seed):
    """Fix all random number generators so runs are repeatable (GPU runs can still vary slightly)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate_model(model, data):
    """Ranking metrics of a fitted model (anything with .intents and .score) on a DataFrame."""
    index = {intent: j for j, intent in enumerate(model.intents)}
    true_idx = data["intent_id"].map(index).to_numpy()
    return ranking_metrics(model.score(data["question"].tolist()), true_idx)


# ----------------------------------------------------------------------------
# Bi-encoder (MNRL)
# ----------------------------------------------------------------------------
def build_pairs(data, train, answer_bank, pair_types, rng):
    """Make (anchor, positive, intent) pairs. Positives always come from TRAIN data / the answer bank.

    `data` is the set of anchors (train questions for training, val questions for val loss).
    A new random positive is drawn every time this is called (i.e. every epoch).
    """
    train_by_intent = train.groupby("intent_id")["question"].apply(list).to_dict()
    answers_by_intent = answer_bank.groupby("intent_id")["simple_answer"].apply(list).to_dict()
    pairs = []
    for question, intent in zip(data["question"], data["intent_id"]):
        if "q2q" in pair_types:
            others = [q for q in train_by_intent.get(intent, []) if q != question]
            if others:  # an intent with a single training question has no partner
                pairs.append((question, others[rng.integers(len(others))], intent))
        if "q2a" in pair_types and intent in answers_by_intent:
            answers = answers_by_intent[intent]
            pairs.append((question, answers[rng.integers(len(answers))], intent))
    return pairs


def unique_intent_batches(pairs, batch_size, rng):
    """Shuffle pairs into batches in which no two pairs share an intent (avoids false negatives)."""
    queues = {}
    for pair in rng.permutation(len(pairs)):
        queues.setdefault(pairs[pair][2], []).append(pairs[pair])
    batches = []
    while queues:
        # take one pair from each of up to `batch_size` randomly chosen intents
        intents = list(queues)
        chosen = rng.choice(len(intents), size=min(batch_size, len(intents)), replace=False)
        batch = [queues[intents[c]].pop() for c in chosen]
        queues = {k: v for k, v in queues.items() if v}
        if len(batch) > 1:  # a batch of 1 has no negatives, so it teaches nothing
            batches.append(batch)
    return batches


def embed(model, texts, device):
    """Sentence embeddings WITH gradients (model.encode() switches gradients off)."""
    features = model.preprocess(texts)  # tokenise: text -> token ids + attention mask
    # move the tensors to the GPU/CPU (some entries are plain metadata, not tensors)
    features = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in features.items()}
    return model(features)["sentence_embedding"]


def mnrl_loss(model, batch, device):
    """MultipleNegativesRankingLoss for one batch of (anchor, positive, intent) pairs."""
    anchors = F.normalize(embed(model, [p[0] for p in batch], device), dim=-1)
    positives = F.normalize(embed(model, [p[1] for p in batch], device), dim=-1)
    similarity = anchors @ positives.T * MNRL_SCALE         # N x N matrix
    targets = torch.arange(len(batch), device=device)       # anchor i belongs with positive i
    return F.cross_entropy(similarity, targets)


def train_bi_encoder(base_model, train, val, answer_bank, pair_types=("q2q", "q2a"), eval_mode="q2q",
                     batch_size=32, lr=2e-5, max_epochs=10, patience=2, warmup_ratio=0.1, max_seq_length=256,
                     seed=42, verbose=True):
    """Fine-tune `base_model` with MNRL. Returns (best model, per-epoch history DataFrame)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()
    model = SentenceTransformer(base_model, device=device)
    # Longer texts are cut at `max_seq_length` tokens. Memory grows with the square of the length,
    # so this keeps bigger models (e.g. PubMedBERT, default 512) within GPU memory.
    # Questions are short (median 7 words) and most answers fit easily.
    model.max_seq_length = min(model.max_seq_length, max_seq_length)

    # Val pairs are built once so val loss is comparable between epochs
    val_pairs = build_pairs(val, train, answer_bank, pair_types, np.random.default_rng(seed))
    val_batches = unique_intent_batches(val_pairs, batch_size, np.random.default_rng(seed))

    steps_per_epoch = len(unique_intent_batches(build_pairs(train, train, answer_bank, pair_types, rng), batch_size, rng))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = steps_per_epoch * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)

    def val_accuracy():
        retriever = EmbeddingRetriever(model, mode=eval_mode).fit(train["question"], train["intent_id"], answer_bank)
        return evaluate_model(retriever, val)["accuracy@1"]

    history = [{"epoch": 0, "train_loss": None, "val_loss": None, "val_accuracy@1": val_accuracy(), "seconds": 0.0}]
    best_acc, best_state, bad_epochs = history[0]["val_accuracy@1"], copy.deepcopy(model.state_dict()), 0
    if verbose:
        print(f"epoch 0 (before training): val acc@1 = {best_acc:.3f}")

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        train_losses = []
        for batch in unique_intent_batches(build_pairs(train, train, answer_bank, pair_types, rng), batch_size, rng):
            loss = mnrl_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_loss = float(np.mean([mnrl_loss(model, b, device).item() for b in val_batches]))
        acc = val_accuracy()
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss,
                        "val_accuracy@1": acc, "seconds": round(time.time() - start, 1)})
        if verbose:
            print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f} | val loss {val_loss:.4f} | "
                  f"val acc@1 {acc:.3f} | {history[-1]['seconds']}s")

        # Early stopping: keep the best weights, stop after `patience` epochs without improvement
        if acc > best_acc:
            best_acc, best_state, bad_epochs = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


# ----------------------------------------------------------------------------
# Transformer classifier (cross-entropy)
# ----------------------------------------------------------------------------
def train_classifier(base_model, train, val, batch_size=16, lr=5e-5, max_epochs=15, patience=3,
                     warmup_ratio=0.1, max_length=128, seed=42, verbose=True):
    """Fine-tune a Transformer classifier. Returns (TransformerClassifier, history DataFrame)."""
    set_seed(seed)
    rng = np.random.default_rng(seed)
    device = get_device()

    intents = sorted(train["intent_id"].unique())
    label_of = {intent: j for j, intent in enumerate(intents)}
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=len(intents),
        id2label=dict(enumerate(intents)), label2id=label_of,
    ).to(device)

    questions = train["question"].tolist()
    labels = train["intent_id"].map(label_of).to_numpy()
    val_labels = torch.tensor(val["intent_id"].map(label_of).to_numpy(), device=device)

    steps_per_epoch = int(np.ceil(len(questions) / batch_size))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = steps_per_epoch * max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)
    wrapper = TransformerClassifier(model, tokenizer, intents, max_length=max_length)

    history, best_acc, best_state, bad_epochs = [], -1.0, None, 0
    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        train_losses = []
        order = rng.permutation(len(questions))  # new shuffle every epoch
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = tokenizer([questions[j] for j in idx], padding=True, truncation=True,
                              max_length=max_length, return_tensors="pt").to(device)
            out = model(**batch, labels=torch.tensor(labels[idx], device=device))  # out.loss = cross-entropy
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            train_losses.append(out.loss.item())

        probs = torch.tensor(wrapper.score(val["question"]), device=device)
        val_loss = F.nll_loss(torch.log(probs + 1e-12), val_labels).item()
        acc = (probs.argmax(dim=1) == val_labels).float().mean().item()
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss,
                        "val_accuracy@1": acc, "seconds": round(time.time() - start, 1)})
        if verbose:
            print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f} | val loss {val_loss:.4f} | "
                  f"val acc@1 {acc:.3f} | {history[-1]['seconds']}s")

        if acc > best_acc:
            best_acc, best_state, bad_epochs = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"early stopping: no improvement for {patience} epochs")
                break

    model.load_state_dict(best_state)
    return wrapper, pd.DataFrame(history)
