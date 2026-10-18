"""
Reads the HASH Excel file into a single DataFrame.

The file has one sheet per category, all with the same 6 columns: intent label, an unused column,
two question columns (crowdsourced question / another way of asking), simple answer, long answer.
The intent label is only written on the first row of each intent block, so it gets forward-filled.
No cleaning happens here, that is all in data_prep.py.
"""

import pandas as pd

RAW_FILE = "data/raw/hash_sti_qa.xlsx"

COLUMN_NAMES = ["intent", "unnamed", "q_crowd", "q_other", "simple_answer", "long_answer"]

# words expected in the original headers, so a changed file layout fails loudly
EXPECTED_HEADER_WORDS = ["intent", "unnamed", "crowdsourcing", "another", "simple", "long"]


def load_raw_sheets(path=RAW_FILE):
    """All sheets stacked, plus category, excel_row and intent_block columns."""
    sheets = pd.read_excel(path, sheet_name=None)

    frames = []
    for sheet_name, df in sheets.items():
        for header, word in zip(df.columns, EXPECTED_HEADER_WORDS):
            if word not in str(header).lower():
                raise ValueError(f"Sheet '{sheet_name}': unexpected header '{header}' (expected '{word}')")

        df = df.copy()
        df.columns = COLUMN_NAMES
        df.insert(0, "category", sheet_name)
        df.insert(1, "excel_row", range(2, len(df) + 2))  # row number as seen in Excel
        df["intent_block"] = df["intent"].ffill()
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def questions_long(raw):
    """One row per question cell, from both question columns."""
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
