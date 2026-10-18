"""
Neural models, same fit()/score() interface as baselines.py.

EmbeddingRetriever: embeds texts with a sentence-transformers model and matches a question to the
closest known question (q2q) or answer (q2a) by cosine similarity. Used for the frozen and the
fine-tuned bi-encoders.
TransformerClassifier: prediction wrapper for the fine-tuned DistilBERT classifier (trained in train.py).
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_device():
    """CUDA on Colab, MPS on my Mac, otherwise CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingRetriever:
    def __init__(self, model, mode="q2q", batch_size=64):
        # model can be a name/path or an already loaded SentenceTransformer
        self.model = SentenceTransformer(model, device=get_device()) if isinstance(model, str) else model
        self.mode = mode
        self.batch_size = batch_size

    def encode(self, texts):
        # normalised vectors, so a dot product is the cosine similarity
        return self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def fit(self, questions, intents, answer_bank=None):
        """Encode the texts that new questions are compared against."""
        self.intents = sorted(set(intents))
        intent_index = {intent: j for j, intent in enumerate(self.intents)}

        if self.mode == "q2q":
            index_texts, index_intents = list(questions), list(intents)
        elif self.mode == "q2a":
            # only answers of intents seen in training
            bank = answer_bank[answer_bank["intent_id"].isin(intent_index)]
            index_texts, index_intents = bank["simple_answer"].tolist(), bank["intent_id"].tolist()
        else:
            raise ValueError(f"unknown mode {self.mode}")

        self.index_intent = np.array([intent_index[i] for i in index_intents])
        self.index_vectors = self.encode(index_texts)
        return self

    def save_index(self, path):
        """Saves only the vectors and intent ids, so the app doesn't have to re-encode on start-up."""
        np.savez(path, vectors=self.index_vectors, index_intent=self.index_intent, intents=np.array(self.intents))

    def load_index(self, path):
        """Load an index saved with save_index (used by the deployed app)."""
        data = np.load(path, allow_pickle=False)
        self.index_vectors, self.index_intent = data["vectors"], data["index_intent"]
        self.intents = data["intents"].tolist()
        return self

    def score_with_similarities(self, questions):
        """Intent scores (best matching text per intent) and the raw similarities."""
        sims = self.encode(questions) @ self.index_vectors.T
        intent_scores = np.full((len(questions), len(self.intents)), -1.0)
        for j in range(len(self.intents)):
            members = self.index_intent == j
            if members.any():
                intent_scores[:, j] = sims[:, members].max(axis=1)
        return intent_scores, sims

    def score(self, questions):
        return self.score_with_similarities(questions)[0]


class TransformerClassifier:
    """Wraps the fine-tuned classifier so it returns a probability matrix like the other models."""

    def __init__(self, model, tokenizer, intents, max_length=128, batch_size=64):
        self.model, self.tokenizer = model, tokenizer
        self.intents = list(intents)
        self.max_length, self.batch_size = max_length, batch_size

    @torch.no_grad()
    def score(self, questions):
        self.model.eval()
        device = next(self.model.parameters()).device
        probabilities = []
        questions = list(questions)
        for start in range(0, len(questions), self.batch_size):
            batch = self.tokenizer(questions[start:start + self.batch_size], padding=True, truncation=True,
                                   max_length=self.max_length, return_tensors="pt").to(device)
            logits = self.model(**batch).logits
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probabilities)
