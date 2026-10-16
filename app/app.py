"""
app.py - Streamlit web interface for the STI question-answering assistant.

Run locally (after `python -m src.build_app`):   streamlit run app/app.py
On Streamlit Community Cloud: deploy this repository with main file `app/app.py`.

Where the model files come from:
  - locally: the app_artifacts/ folder built by src/build_app.py;
  - on the cloud: the same folder, downloaded once from a Hugging Face model repository
    (set STI_QA_HF_REPO to change which one).

The app only displays what src/inference.py returns. It does not log, store or print questions
(sensitive topic; users may be teenagers), and Streamlit's usage statistics are switched off
in .streamlit/config.toml.
"""

import os
import sys

import pandas as pd
import streamlit as st

# Make `src` importable: the project root is one level above app/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import DISCLAIMER, STIAssistant  # noqa: E402

HF_REPO = os.environ.get("STI_QA_HF_REPO", "EddyIrasetsa/sti-qa-assistant")
LOCAL_ARTIFACTS = os.path.join(ROOT, "app_artifacts")

EXAMPLES = [
    "Can I get syphilis from kissing?",
    "How soon after unprotected sex should I test for HIV?",
    "Is genital herpes curable?",
    "What are the signs of gonorrhoea in women?",
    "Can hepatitis B pass from a mother to her baby?",
    "Does washing after sex prevent STIs?",
    "How do I plant maize?",  # out-of-domain example: the assistant should refuse
]


@st.cache_resource(show_spinner="Loading the models (the first start can take a minute)...")
def load_assistant():
    """Load both models and the answers once; Streamlit keeps them in memory for every visitor."""
    if os.path.exists(os.path.join(LOCAL_ARTIFACTS, "config.json")):
        path = LOCAL_ARTIFACTS
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(HF_REPO, repo_type="model")
    return STIAssistant(path)


st.set_page_config(page_title="STI Q&A Assistant", page_icon="🩺")
st.title("STI Question & Answer Assistant")
st.markdown(
    "Ask a question about sexually transmitted infections in everyday English. The assistant finds the most "
    "similar question in its knowledge base and shows an answer **written by medical professionals** "
    "(HASH STI Q&A dataset, Uganda). It does not generate text, and it tells you when a question is outside "
    "its topics."
)
st.warning(DISCLAIMER, icon="⚠️")

assistant = load_assistant()

example = st.selectbox("Try an example (optional)", [""] + EXAMPLES)
with st.form("ask"):
    question = st.text_area("Your question", value=example, height=90,
                            placeholder="e.g. Can I get chlamydia from a toilet seat?")
    detail = st.radio("Answer type", ["Simple", "Detailed"], horizontal=True)
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    result = assistant.answer(question, detail.lower())
    if result["status"] == "empty":
        st.info(result["message"])
    else:
        if result["status"] == "out_of_domain":
            st.error("**I can't answer that.** " + result["message"])
        else:
            if result["status"] == "not_sure":
                st.info(result["message"])
            else:
                st.success(f"**Matched topic:** {result['category']} – {result['topic']}")
            st.markdown(result["answer"])

        st.caption(f"Confidence (cosine similarity): {result['confidence']:.2f} · answer at ≥ "
                   f"{result['threshold']:.2f} · \"not sure\" at ≥ {result['low_threshold']:.2f} · "
                   f"below that: no answer")
        st.markdown("**Top 3 matched topics**")
        st.dataframe(pd.DataFrame(result["top_3"]).rename(columns=str.capitalize),
                     hide_index=True, width="stretch")

st.divider()
st.markdown(
    "**Topics covered:** general STIs, HIV, syphilis, gonorrhoea, chlamydia, trichomoniasis, genital herpes, "
    "hepatitis B.  \n"
    "**Privacy:** your questions are not stored or logged.  \n"
    "**Data:** Dataset Club, HASH (2026). *The HASH Sexually Transmitted Infection Question and Answer "
    "Crowdsourced Dataset*. Harvard Dataverse, https://doi.org/10.7910/DVN/5SWM5U, licensed CC BY-NC 4.0. "
    "Models: fine-tuned from NeuML/pubmedbert-base-embeddings, plus sentence-transformers/all-MiniLM-L6-v2. "
    "Non-commercial student project."
)
