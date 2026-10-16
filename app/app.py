"""
app.py - Gradio web interface for the STI question-answering assistant.

Run locally (after `python -m src.build_app`):   python app/app.py
On Hugging Face Spaces this file is copied to the Space's root by src/build_app.py.

The app only displays what src/inference.py returns. It does not log, store or print questions
(sensitive topic; users may be teenagers), and Gradio's usage analytics are switched off.
"""

import os
import sys

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # no usage analytics sent to Gradio

import gradio as gr

# Make `src` importable both locally (app/ is one level below the project) and on Spaces (same level)
HERE = os.path.dirname(os.path.abspath(__file__))
for path in (HERE, os.path.dirname(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.inference import DISCLAIMER, STIAssistant  # noqa: E402

# Artifacts: next to app.py on Spaces, or in space_build/ when running from the project folder
CANDIDATES = [os.environ.get("STI_QA_ARTIFACTS", ""), os.path.join(HERE, "artifacts"),
              os.path.join(os.path.dirname(HERE), "space_build", "artifacts")]
ARTIFACTS_DIR = next(p for p in CANDIDATES if p and os.path.exists(os.path.join(p, "config.json")))
assistant = STIAssistant(ARTIFACTS_DIR)

EXAMPLES = [
    "Can I get syphilis from kissing?",
    "How soon after unprotected sex should I test for HIV?",
    "Is genital herpes curable?",
    "What are the signs of gonorrhoea in women?",
    "Can hepatitis B pass from a mother to her baby?",
    "Does washing after sex prevent STIs?",
    "How do I plant maize?",  # out-of-domain example: the assistant should refuse
]


def respond(question, detail):
    """Called when the user clicks Ask. Returns (answer markdown, topic markdown, top-3 table)."""
    result = assistant.answer(question, "detailed" if detail == "Detailed" else "simple")
    if result["status"] == "empty":
        return f"*{result['message']}*", "", []

    table = [[t["topic"], t["category"], t["similarity"]] for t in result["top_3"]]
    confidence_line = (f"**Confidence (cosine similarity):** {result['confidence']:.2f} "
                       f"(answer at ≥ {result['threshold']:.2f}, not sure at ≥ {result['low_threshold']:.2f})")
    if result["status"] == "out_of_domain":
        answer_md = f"### I can't answer that\n{result['message']}"
        topic_md = confidence_line
    else:
        title = "### Answer" if result["status"] == "answer" else f"### Not sure\n{result['message']}\n\n---\n"
        answer_md = f"{title}\n{result['answer']}"
        topic_md = f"**Matched topic:** {result['category']} – {result['topic']}  \n{confidence_line}"
    return answer_md, topic_md, table


with gr.Blocks(title="STI Q&A Assistant", analytics_enabled=False) as demo:
    gr.Markdown(
        "# STI Question & Answer Assistant\n"
        "Ask a question about sexually transmitted infections in everyday English. The assistant finds the "
        "most similar question in its knowledge base and shows an answer **written by medical professionals** "
        "(HASH STI Q&A dataset, Uganda). It does not generate text, and it says so when a question is outside "
        "its topics."
    )
    gr.Markdown(f"> ⚠️ **{DISCLAIMER}**")

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(label="Your question", lines=3,
                                  placeholder="e.g. Can I get chlamydia from a toilet seat?")
            detail = gr.Radio(["Simple", "Detailed"], value="Simple", label="Answer type")
            ask = gr.Button("Ask", variant="primary")
            gr.Examples(EXAMPLES, inputs=question, label="Try an example")
        with gr.Column(scale=1):
            answer_box = gr.Markdown()
            topic_box = gr.Markdown()
            top3_box = gr.Dataframe(headers=["Topic", "Category", "Similarity"], label="Top 3 matched topics",
                                    interactive=False)

    gr.Markdown(
        "**Topics covered:** general STIs, HIV, syphilis, gonorrhoea, chlamydia, trichomoniasis, genital herpes, "
        "hepatitis B.  \n"
        "**Privacy:** your questions are not stored or logged.  \n"
        "**Data:** Dataset Club, HASH (2026). *The HASH Sexually Transmitted Infection Question and Answer "
        "Crowdsourced Dataset*. Harvard Dataverse, https://doi.org/10.7910/DVN/5SWM5U, licensed CC BY-NC 4.0. "
        "Model: fine-tuned from NeuML/pubmedbert-base-embeddings. Non-commercial student project."
    )

    ask.click(respond, inputs=[question, detail], outputs=[answer_box, topic_box, top3_box])
    question.submit(respond, inputs=[question, detail], outputs=[answer_box, topic_box, top3_box])


if __name__ == "__main__":
    demo.launch()
