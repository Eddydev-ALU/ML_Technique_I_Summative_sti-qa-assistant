"""
baselines.py - Simple, non-neural baselines that the main models must beat.

All three follow the same interface:
    model.fit(train_questions, train_intents)
    scores = model.score(questions)   # matrix: one row per question, one column per intent
The column order is model.intents (sorted list of intent ids seen in training).

1. MajorityBaseline : always ranks intents by how often they appear in training.
                      Shows what you get with no understanding of the question at all.
2. BM25Baseline     : classic keyword search. Find the training questions that share the most
                      (rare) words with the new question; an intent's score is the score of
                      its best-matching training question.
3. TfidfLogRegBaseline : turns each question into a TF-IDF vector (word weights) and trains a
                      logistic regression classifier to predict the intent.
"""

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

SEED = 42


def tokenize(text):
    """Lower-case and split into words (letters and digits only)."""
    return re.findall(r"[a-z0-9]+", text.lower())


class MajorityBaseline:
    def fit(self, questions, intents):
        counts = {}
        for intent in intents:
            counts[intent] = counts.get(intent, 0) + 1
        self.intents = sorted(counts)
        self.prior = np.array([counts[i] for i in self.intents], dtype=float)
        return self

    def score(self, questions):
        # Same scores for every question: the training frequency of each intent
        return np.tile(self.prior, (len(questions), 1))


class BM25Baseline:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # how quickly repeated words stop adding score
        self.b = b    # how much long documents are penalised

    def fit(self, questions, intents):
        self.intents = sorted(set(intents))
        intent_index = {intent: j for j, intent in enumerate(self.intents)}
        self.doc_intent = np.array([intent_index[i] for i in intents])  # intent of each training question
        self.bm25 = BM25Okapi([tokenize(q) for q in questions], k1=self.k1, b=self.b)
        return self

    def score_documents(self, questions):
        """BM25 score of every new question against every training question."""
        return np.array([self.bm25.get_scores(tokenize(q)) for q in questions])

    def score(self, questions):
        doc_scores = self.score_documents(questions)
        # An intent's score = the best score among its training questions
        intent_scores = np.zeros((len(questions), len(self.intents)))
        for j in range(len(self.intents)):
            intent_scores[:, j] = doc_scores[:, self.doc_intent == j].max(axis=1)
        return intent_scores


class TfidfLogRegBaseline:
    def __init__(self, analyzer="word", ngram_range=(1, 2), C=10.0):
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,        # "word" = words, "char_wb" = character pieces inside words
            ngram_range=ngram_range,  # e.g. (1, 2) = single words and pairs of words
            lowercase=True,
            sublinear_tf=True,        # dampen repeated words: 1 + log(count)
        )
        # C = inverse regularisation strength: small C = simpler model, large C = fits training data harder
        self.classifier = LogisticRegression(C=C, max_iter=5000, random_state=SEED)

    def fit(self, questions, intents):
        X = self.vectorizer.fit_transform(questions)  # vocabulary learned from TRAIN only
        self.classifier.fit(X, intents)
        self.intents = list(self.classifier.classes_)  # sorted intent ids
        return self

    def score(self, questions):
        # Predicted probability of each intent
        return self.classifier.predict_proba(self.vectorizer.transform(questions))
