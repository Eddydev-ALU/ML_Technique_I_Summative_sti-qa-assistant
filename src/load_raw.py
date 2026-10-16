"""
load_raw.py - Read the HASH STI Q&A Excel file into one pandas DataFrame.

The Excel file (data/raw/hash_sti_qa.xlsx, see src/download_data.py) has one sheet per
category (General STIs, HIV, Syphilis, ...). Every sheet has the same 6 columns:

    0  <Category> Chatbot Intents      intent label, written only on the FIRST row of each intent block
    1  Unnamed: 1                      almost always empty
    2  Question(s) from Crowdsourcing  a question
    3  Another Questions (...)         another question (a different way to ask)
    4  Simple Answer                   short answer
    5  Long Answer                     detailed answer

An "intent block" starts at a row with an intent label and continues until the next label.
This function only RESHAPES the data (no cleaning, no dropping). Cleaning is done in
src/data_prep.py so that every change to the data is in one place and easy to explain.
"""

import pandas as pd

RAW_FILE = "data/raw/hash_sti_qa.xlsx"

# Short names we give the 6 columns, in order
COLUMN_NAMES = ["intent", "unnamed", "q_crowd", "q_other", "simple_answer", "long_answer"]

# Words we expect in each original header. We check them so that if the file layout
# is ever different, the code stops with a clear error instead of mixing up columns.
EXPECTED_HEADER_WORDS = ["intent", "unnamed", "crowdsourcing", "another", "simple", "long"]


def load_raw_sheets(path=RAW_FILE):
    """Return one DataFrame with all sheets stacked, plus helper columns.

    Added columns:
      category      : sheet name (General STIs, HIV, ...)
      excel_row     : row number as shown in Excel (header is row 1), for tracing back
      intent_block  : intent label forward-filled down its block (so every row knows its intent)
    """
    sheets = pd.read_excel(path, sheet_name=None)  # dict: {sheet name: DataFrame}

    frames = []
    for sheet_name, df in sheets.items():
        # Check the header before renaming
        for header, word in zip(df.columns, EXPECTED_HEADER_WORDS):
            if word not in str(header).lower():
                raise ValueError(f"Sheet '{sheet_name}': unexpected header '{header}' (expected '{word}')")

        df = df.copy()
        df.columns = COLUMN_NAMES
        df.insert(0, "category", sheet_name)
        df.insert(1, "excel_row", range(2, len(df) + 2))
        # The intent label is only on the first row of a block: copy it down to the rest
        df["intent_block"] = df["intent"].ffill()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def questions_long(raw):
    """One row per question cell (both question columns), with its category and intent block."""
    parts = []
    for col, source in [("q_crowd", "crowdsourced"), ("q_other", "another_way")]:
        part = raw.loc[raw[col].notna(), ["category", "excel_row", "intent_block", col]]
        part = part.rename(columns={col: "question"})
        part["question_source"] = source
        parts.append(part)
    return pd.concat(parts, ignore_index=True).sort_values(["category", "excel_row"], kind="stable")


if __name__ == "__main__":
    raw = load_raw_sheets()
    print(raw.shape)
    print(questions_long(raw).head())
