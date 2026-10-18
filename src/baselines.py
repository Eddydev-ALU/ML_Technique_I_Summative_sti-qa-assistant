"""
Baselines: majority class, BM25 and TF-IDF + logistic regression.

Same interface as the neural models: fit(questions, intents), then score(questions) returns a
matrix with one row per question and one column per intent (column order = self.intents).
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class MajorityBaseline:
    """Ranks intents by how often they appear in training, ignoring the question."""

    def fit(self, questions, intents):
        counts = {}
        for intent in intents:
            counts[intent] = counts.get(intent, 0) + 1
        self.intents = sorted(counts)
        self.prior = np.array([counts[i] for i in self.intents], dtype=float)
        return self

    def score(self, questions):
        return np.tile(self.prior, (len(questions), 1))


class BM25Baseline:
    """Keyword search over the training questions. Intent score = its best matching question."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # term frequency saturation
        self.b = b    # length normalisation

    def fit(self, questions, intents):
        self.intents = sorted(set(intents))
        intent_index = {intent: j for j, intent in enumerate(self.intents)}
        self.doc_intent = np.array([intent_index[i] for i in intents])
        self.bm25 = BM25Okapi([tokenize(q) for q in questions], k1=self.k1, b=self.b)
        return self

    def score_documents(self, questions):
        return np.array([self.bm25.get_scores(tokenize(q)) for q in questions])

    def score(self, questions):
        doc_scores = self.score_documents(questions)
        intent_scores = np.zeros((len(questions), len(self.intents)))
        for j in range(len(self.intents)):
            intent_scores[:, j] = doc_scores[:, self.doc_intent == j].max(axis=1)
        return intent_scores


class TfidfLogRegBaseline:
    def __init__(self, analyzer="word", ngram_range=(1, 2), C=10.0):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,  # "word" or "char_wb" (character n-grams, handles misspellings)
            ngram_range=ngram_range,
            lowercase=True,
            sublinear_tf=True,
        )
        self.classifier = LogisticRegression(C=C, max_iter=5000, random_state=SEED)

    def fit(self, questions, intents):
        X = self.vectorizer.fit_transform(questions)
        self.classifier.fit(X, intents)
        self.intents = list(self.classifier.classes_)
        return self

    def score(self, questions):
        return self.classifier.predict_proba(self.vectorizer.transform(questions))
