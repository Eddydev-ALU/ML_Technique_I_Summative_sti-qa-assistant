"""
Streamlit interface for the STI Q&A assistant.

Local: python -m src.build_app, then streamlit run app/app.py
Streamlit Cloud: main file app/app.py. The model files are downloaded from the Hugging Face repo
on start-up (STI_QA_HF_REPO). Questions are never stored or logged, and usage stats are turned
off in .streamlit/config.toml.
"""

import os
import sys

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import DISCLAIMER, STIAssistant  # noqa: E402

HF_REPO = os.environ.get("STI_QA_HF_REPO", "EddyIrasetsa/sti-qa-assistant")
LOCAL_ARTIFACTS = os.path.join(ROOT, "app_artifacts")

# checked by hand: confident and correct answers, covering all 8 categories
EXAMPLES = [
    "Can I get syphilis from kissing?",
    "Is there a cure for HIV?",
    "What are the symptoms of HIV?",
    "How can I prevent HIV?",
    "Can I get an STI from a toilet seat?",
    "What is gonorrhea?",
    "Can chlamydia be cured?",
    "How is trichomoniasis treated?",
    "Is there a cure for herpes?",
    "Is there a vaccine for hepatitis B?",
    "How is hepatitis B spread?",
]
OFF_TOPIC_EXAMPLE = "How do I plant maize?"


@st.cache_resource(show_spinner="Loading the models (the first start can take a minute)...")
def load_assistant():
    """Loaded once and shared between visitors."""
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

example = st.selectbox(
    "Try an example (optional)", [""] + EXAMPLES + [OFF_TOPIC_EXAMPLE],
    format_func=lambda q: f"{q}  (off-topic: shows a refusal)" if q == OFF_TOPIC_EXAMPLE else q,
)
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
