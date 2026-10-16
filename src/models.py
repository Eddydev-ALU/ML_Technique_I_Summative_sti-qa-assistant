"""
models.py - The neural models, with the same fit()/score() interface as src/baselines.py.

1. EmbeddingRetriever
   A sentence-embedding model (a Transformer that turns a sentence into a vector of numbers,
   e.g. 384 numbers for MiniLM) is used to find the nearest known text:
     mode="q2q": compare the user's question with every TRAINING QUESTION
     mode="q2a": compare the user's question with every ANSWER in the answer bank
   Similarity = cosine similarity of the vectors (1 = same direction/meaning, 0 = unrelated).
   An intent's score = the similarity of its best-matching question (or answer).
   The same class is used for the frozen model and for our fine-tuned model.

2. TransformerClassifier
   A Transformer (DistilBERT) with a classification layer on top that outputs one probability
   per intent. Trained in src/train.py; this class only does prediction.
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def get_device():
    """Use a GPU if there is one: CUDA (Colab) or MPS (Apple Silicon), else the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class EmbeddingRetriever:
    def __init__(self, model, mode="q2q", batch_size=64):
        # `model` is a model name/path, or an already-loaded SentenceTransformer
        self.model = SentenceTransformer(model, device=get_device()) if isinstance(model, str) else model
        self.mode = mode
        self.batch_size = batch_size

    def encode(self, texts):
        # normalize_embeddings=True makes every vector length 1, so dot product = cosine similarity
        return self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)

    def fit(self, questions, intents, answer_bank=None):
        """Build the index: the texts we compare new questions against."""
        self.intents = sorted(set(intents))
        intent_index = {intent: j for j, intent in enumerate(self.intents)}

        if self.mode == "q2q":
            index_texts, index_intents = list(questions), list(intents)
        elif self.mode == "q2a":
            # Only answers of intents the model is allowed to know (held-out intents stay unknown)
            bank = answer_bank[answer_bank["intent_id"].isin(intent_index)]
            index_texts, index_intents = bank["simple_answer"].tolist(), bank["intent_id"].tolist()
        else:
            raise ValueError(f"unknown mode {self.mode}")

        self.index_intent = np.array([intent_index[i] for i in index_intents])
        self.index_vectors = self.encode(index_texts)
        return self

    def save_index(self, path):
        """Save the index vectors (numbers only, no text) so the app does not re-encode at start-up."""
        np.savez(path, vectors=self.index_vectors, index_intent=self.index_intent, intents=np.array(self.intents))

    def load_index(self, path):
        """Load an index saved with save_index (used by the deployed app)."""
        data = np.load(path, allow_pickle=False)
        self.index_vectors, self.index_intent = data["vectors"], data["index_intent"]
        self.intents = data["intents"].tolist()
        return self

    def score_with_similarities(self, questions):
        """Return (intent score matrix, similarity to every index text)."""
        sims = self.encode(questions) @ self.index_vectors.T  # cosine similarity to every index text
        intent_scores = np.full((len(questions), len(self.intents)), -1.0)
        for j in range(len(self.intents)):
            members = self.index_intent == j
            if members.any():
                intent_scores[:, j] = sims[:, members].max(axis=1)
        return intent_scores, sims

    def score(self, questions):
        return self.score_with_similarities(questions)[0]


class TransformerClassifier:
    """Prediction wrapper around a fine-tuned sequence-classification model."""

    def __init__(self, model, tokenizer, intents, max_length=128, batch_size=64):
        self.model, self.tokenizer = model, tokenizer
        self.intents = list(intents)  # column order of the score matrix
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
