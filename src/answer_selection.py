"""
Picks one answer inside an intent. 61 of the 110 intents have more than one answer
(e.g. syphilis transmission has separate answers for toilets, kissing, genetics...).

"first": always the intent's first answer. "semantic": the answer whose text is most similar
to the question. Both are compared in notebook 04.
"""

import numpy as np


class AnswerSelector:
    def __init__(self, encode, answer_bank, answer_vectors=None):
        # answer_vectors can be passed in precomputed (the app does this)
        self.encode = encode
        self.bank = answer_bank.reset_index(drop=True)
        self.answer_vectors = (answer_vectors if answer_vectors is not None
                               else encode(self.bank["simple_answer"].tolist()))
        self.rows_of_intent = self.bank.groupby("intent_id").indices

    def select(self, questions, intents, method="semantic"):
        """answer_id for each (question, intent) pair."""
        question_vectors = self.encode(list(questions)) if method == "semantic" else None
        chosen = []
        for i, intent in enumerate(intents):
            rows = self.rows_of_intent[intent]
            if method == "first" or len(rows) == 1:
                best_row = rows[0]  # bank keeps the Excel order
            else:
                similarity = self.answer_vectors[rows] @ question_vectors[i]
                best_row = rows[int(np.argmax(similarity))]
            chosen.append(self.bank.at[best_row, "answer_id"])
        return chosen
