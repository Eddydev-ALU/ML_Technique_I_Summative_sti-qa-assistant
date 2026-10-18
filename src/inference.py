"""
Answers a single question with the deployed system (used by app/app.py).

1. Topic: the fine-tuned PubMedBERT compares the question with all 2,502 known questions.
   Confidence = cosine similarity of the closest one.
2. Decision: >= threshold -> answer, between the two thresholds -> "not sure, did you mean...",
   below low_threshold -> refuse. Both thresholds were tuned in notebook 04.
3. Answer inside the topic, with a frozen MiniLM: the paired answer if the question is almost
   identical to a question that sits next to an answer in the Excel file, otherwise the answer with
   the most similar text. Frozen because the fine-tuned model pulls all questions of an intent
   together and can't tell them apart anymore (notebook 04: 0.556 vs 0.465).

Questions are not stored or logged anywhere.
Expects the folder made by src/build_app.py (models, embeddings, answer_bank.csv, config.json).
"""

import json
import os

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.answer_selection import AnswerSelector
from src.models import EmbeddingRetriever, get_device

DISCLAIMER = ("Educational information only, not a diagnosis. "
              "Please visit a health facility for testing and treatment.")
OUT_OF_DOMAIN_MESSAGE = (
    "Sorry, I can only answer questions about sexually transmitted infections (STIs): general STIs, HIV, "
    "syphilis, gonorrhoea, chlamydia, trichomoniasis, genital herpes and hepatitis B. "
    "Your question seems to be outside what I was trained on. Try rephrasing it, or ask a health worker."
)
NOT_SURE_MESSAGE = ("I'm not sure I understood your question. Did you mean a question about **{topic}**? "
                    "If so, here is the answer. If not, please try rephrasing.")

NEAR_IDENTICAL = 0.95  # cosine similarity with the frozen MiniLM

# removed from intent names for display, the category is shown separately anyway
_DISEASE_WORDS = {"stis", "sti", "hiv", "gonorrhea", "chlamydia", "syphilis", "trichomoniasis", "hepatitis",
                  "herpes", "simplex", "virus", "infections"}


def readable_intent(intent_id):
    """'Herpes::Transmission_spread_infections' -> 'Transmission / spread'"""
    label = intent_id.split("::", 1)[-1].replace("/", " / ").replace("_", " ")
    words = [w for w in label.split() if w.lower() not in _DISEASE_WORDS]
    if words and words[-1].lower() in ("yes", "no"):
        words = words[:-1] + [f"({words[-1].lower()})"]
    text = " ".join(words).replace(" / ", "/").replace("/", " / ").strip(" /")
    return (text[:1].upper() + text[1:]) if text else intent_id


class STIAssistant:
    def __init__(self, artifacts_dir, device=None):
        device = device or get_device()
        with open(os.path.join(artifacts_dir, "config.json")) as f:
            self.config = json.load(f)
        self.threshold = self.config["threshold"]
        self.low_threshold = self.config["low_threshold"]

        # topic model + index of known questions
        step1 = SentenceTransformer(os.path.join(artifacts_dir, "model"), device=device)
        self.retriever = EmbeddingRetriever(step1, mode="q2q").load_index(os.path.join(artifacts_dir, "index.npz"))
        self.intents = np.array(self.retriever.intents)

        # answer model + paired questions + answers
        self.step2 = EmbeddingRetriever(SentenceTransformer(os.path.join(artifacts_dir, "step2_model"), device=device))
        paired = np.load(os.path.join(artifacts_dir, "paired.npz"))
        self.paired_vectors, self.paired_answer_ids = paired["vectors"], paired["answer_ids"]
        self.bank = pd.read_csv(os.path.join(artifacts_dir, "answer_bank.csv"), keep_default_na=False)
        self.answer_intent = dict(zip(self.bank["answer_id"], self.bank["intent_id"]))
        self.paired_intents = np.array([self.answer_intent[a] for a in self.paired_answer_ids])
        answer_vectors = np.load(os.path.join(artifacts_dir, "answers.npz"))["vectors"]
        self.selector = AnswerSelector(self.step2.encode, self.bank, answer_vectors=answer_vectors)
        self.category_of = dict(zip(self.bank["intent_id"], self.bank["category"]))

    def choose_answer(self, question, intent):
        """Returns (answer_id, how it was chosen)."""
        question_vector = self.step2.encode([question])[0]
        rows = np.where(self.paired_intents == intent)[0]
        if len(rows):
            similarities = self.paired_vectors[rows] @ question_vector
            best = int(np.argmax(similarities))
            if similarities[best] >= NEAR_IDENTICAL:
                return str(self.paired_answer_ids[rows[best]]), "answer paired with a near-identical question"
        return self.selector.select([question], [intent], method="semantic")[0], "most similar answer text"

    def answer(self, question, detail="simple"):
        """detail is "simple" or "detailed". status is answer / not_sure / out_of_domain / empty."""
        question = " ".join(str(question).split())
        if not question:
            return {"status": "empty", "answer": None, "message": "Please type a question.",
                    "top_3": [], "disclaimer": DISCLAIMER}

        # 1. topic
        scores = self.retriever.score([question])[0]
        order = np.argsort(-scores)
        top_3 = [{"topic": readable_intent(self.intents[j]),
                  "category": self.category_of.get(self.intents[j], self.intents[j].split("::")[0]),
                  "similarity": round(float(scores[j]), 3)} for j in order[:3]]
        confidence = float(scores[order[0]])
        best_intent = str(self.intents[order[0]])
        result = {"confidence": round(confidence, 3), "threshold": round(self.threshold, 3),
                  "low_threshold": round(self.low_threshold, 3), "top_3": top_3, "disclaimer": DISCLAIMER}

        # 2. answer, not sure or refuse
        if confidence < self.low_threshold:
            return {**result, "status": "out_of_domain", "answer": None, "message": OUT_OF_DOMAIN_MESSAGE}
        status = "answer" if confidence >= self.threshold else "not_sure"

        # 3. which answer
        answer_id, how = self.choose_answer(question, best_intent)
        row = self.bank.loc[self.bank["answer_id"] == answer_id].iloc[0]
        text = row["long_answer"] if detail == "detailed" else row["simple_answer"]
        return {**result, "status": status, "answer": text,
                "message": (NOT_SURE_MESSAGE.format(topic=f"{row['category']} – {readable_intent(best_intent).lower()}")
                            if status == "not_sure" else ""),
                "intent_id": best_intent, "topic": readable_intent(best_intent), "category": row["category"],
                "answer_id": answer_id, "answer_selection": how}


_default_assistant = None


def answer(question, detail="simple", artifacts_dir="app_artifacts"):
    global _default_assistant
    if _default_assistant is None:
        _default_assistant = STIAssistant(artifacts_dir)
    return _default_assistant.answer(question, detail)
