"""
answer_selection.py - Step 2 of the system: pick ONE answer inside the predicted intent.

49 intents have a single answer, so step 2 is trivial for them. The other 61 intents hold several
answers (e.g. Syphilis transmission: "not from toilets", "rarely from kissing", "not genetic", ...).
For those we compare the user's question with each candidate answer's text using the same
embedding model, and return the most similar answer.

Methods (compared in notebooks/04_error_analysis.ipynb):
  "first"    : always the first answer of the intent (what a simple FAQ bot would do)
  "semantic" : the answer whose text is most similar to the question
"""

import numpy as np


class AnswerSelector:
    def __init__(self, encode, answer_bank, answer_vectors=None):
        # `encode` turns a list of texts into normalised vectors (e.g. EmbeddingRetriever.encode).
        # `answer_vectors` can be passed in pre-computed (the app does this to start faster).
        self.encode = encode
        self.bank = answer_bank.reset_index(drop=True)
        self.answer_vectors = (answer_vectors if answer_vectors is not None
                               else encode(self.bank["simple_answer"].tolist()))
        self.rows_of_intent = self.bank.groupby("intent_id").indices  # intent -> row numbers in the bank

    def select(self, questions, intents, method="semantic"):
        """Return the chosen answer_id for each (question, intent) pair."""
        question_vectors = self.encode(list(questions)) if method == "semantic" else None
        chosen = []
        for i, intent in enumerate(intents):
            rows = self.rows_of_intent[intent]
            if method == "first" or len(rows) == 1:
                best_row = rows[0]  # bank rows keep the Excel order, so rows[0] is the first answer
            else:
                similarity = self.answer_vectors[rows] @ question_vectors[i]
                best_row = rows[int(np.argmax(similarity))]
            chosen.append(self.bank.at[best_row, "answer_id"])
        return chosen
