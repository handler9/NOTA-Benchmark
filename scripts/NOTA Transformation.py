#!/usr/bin/env python3
import pandas as pd
import random
import re

# ----------------------------
# Config
# ----------------------------
INPUT_PATH = "afrimedqa_490_hard_mcq_with_options.csv"   # <-- rename to your actual file name
OUTPUT_PATH = "afrimedqa_490_NOTA_seed42.csv"
SEED = 42

TOTAL_ROWS = 490
N_NOTA = 245   # 245 NOTA / 245 INTACT

random.seed(SEED)

# ----------------------------
# Load
# ----------------------------
df = pd.read_csv(INPUT_PATH)

required = [
    "sample_id", "question",
    "option_A", "option_B", "option_C", "option_D", "option_E",
    "correct_answer", "specialty"
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

if len(df) != TOTAL_ROWS:
    raise ValueError(f"Expected {TOTAL_ROWS} rows, found {len(df)}")

# ----------------------------
# Normalize correct answer (A–E only)
# ----------------------------
def normalize_correct_answer(raw, row):
    """
    Returns one of {'A','B','C','D','E'} or raises with a helpful message.
    Handles:
      - 'A', 'a', 'Option A', 'Answer: B', 'B.' etc.
      - '1'..'5' mapped to A..E
      - full option text (matches against option_A..E)
    """
    raw_s = "" if pd.isna(raw) else str(raw).strip()

    # 1) If it's a number 1-5
    if raw_s.isdigit():
        n = int(raw_s)
        if 1 <= n <= 5:
            return ["A","B","C","D","E"][n-1]

    # 2) Regex for a standalone letter A-E anywhere (Option A, Answer: C, etc.)
    m = re.search(r"\b([A-E])\b", raw_s.upper())
    if m:
        return m.group(1)

    # 3) Try if raw is the full text of the correct option
    options = {
        "A": str(row["option_A"]),
        "B": str(row["option_B"]),
        "C": str(row["option_C"]),
        "D": str(row["option_D"]),
        "E": str(row["option_E"]),
    }
    for k, v in options.items():
        if raw_s.strip() == v.strip():
            return k

    # 4) Give a useful error
    raise ValueError(
        f"Could not normalize correct_answer='{raw_s}' for sample_id={row['sample_id']}. "
        f"Expected A-E / 1-5 / or exact option text."
    )

df["correct_letter"] = df.apply(lambda r: normalize_correct_answer(r["correct_answer"], r), axis=1)

# ----------------------------
# Select NOTA rows (seeded)
# Rule: any original correct == 'E' MUST be NOTA (since output only has A–D)
# Then sample remaining rows to reach N_NOTA
# ----------------------------
e_correct_idx = set(df.index[df["correct_letter"] == "E"].tolist())
if len(e_correct_idx) > N_NOTA:
    raise ValueError(
        f"Too many E-correct questions ({len(e_correct_idx)}) to fit into {N_NOTA} NOTA rows. "
        "Either increase N_NOTA or change rules."
    )

remaining_idx = [i for i in df.index.tolist() if i not in e_correct_idx]
need_more = N_NOTA - len(e_correct_idx)
nota_idx = set(e_correct_idx) | set(random.sample(remaining_idx, need_more))

# ----------------------------
# Transform
# ----------------------------
out_rows = []

for idx, r in df.iterrows():
    correct_letter = r["correct_letter"]

    options = {
        "A": r["option_A"],
        "B": r["option_B"],
        "C": r["option_C"],
        "D": r["option_D"],
        "E": r["option_E"],
    }

    base = {
        "question_id": r["sample_id"],
        "category": r["specialty"],
        "question_type": None,
        "correct_choice": None,
        "stem": r["question"],
        "option_A": None,
        "option_B": None,
        "option_C": None,
        "option_D": None,
    }

    if idx in nota_idx:
        # NOTA: remove the correct option content; A–D are all incorrect
        incorrect_texts = [options[k] for k in ["A","B","C","D","E"] if k != correct_letter]
        if len(incorrect_texts) != 4:
            raise ValueError(
                f"Row idx={idx}, sample_id={r['sample_id']}: expected 4 incorrect options, got {len(incorrect_texts)} "
                f"(correct_letter={correct_letter})"
            )
        random.shuffle(incorrect_texts)

        base.update({
            "question_type": "NOTA",
            "correct_choice": "NOTA",
            "option_A": incorrect_texts[0],
            "option_B": incorrect_texts[1],
            "option_C": incorrect_texts[2],
            "option_D": incorrect_texts[3],
        })
    else:
        # INTACT: must be A–D
        if correct_letter not in ["A","B","C","D"]:
            raise ValueError(
                f"INTACT row idx={idx}, sample_id={r['sample_id']} has correct_letter={correct_letter} (should not happen)."
            )

        base.update({
            "question_type": "INTACT",
            "correct_choice": correct_letter,
            "option_A": options["A"],
            "option_B": options["B"],
            "option_C": options["C"],
            "option_D": options["D"],
        })

    out_rows.append(base)

out = pd.DataFrame(out_rows)

# ----------------------------
# Final shuffle (seeded)
# ----------------------------
out = out.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ----------------------------
# Enforce exact column order
# ----------------------------
out = out[
    ["question_id","category","question_type","correct_choice","stem",
     "option_A","option_B","option_C","option_D"]
]

# ----------------------------
# Sanity checks
# ----------------------------
n_nota = (out["question_type"] == "NOTA").sum()
n_intact = (out["question_type"] == "INTACT").sum()
if n_nota != N_NOTA or n_intact != (TOTAL_ROWS - N_NOTA):
    raise ValueError(f"Split wrong: NOTA={n_nota}, INTACT={n_intact}")

bad_intact = out.query("question_type == 'INTACT' and correct_choice not in ['A','B','C','D']")
if len(bad_intact) > 0:
    raise ValueError("Found INTACT rows with invalid correct_choice (not A–D).")

bad_nota = out.query("question_type == 'NOTA' and correct_choice != 'NOTA'")
if len(bad_nota) > 0:
    raise ValueError("Found NOTA rows where correct_choice != 'NOTA'.")

# ----------------------------
# Save
# ----------------------------
out.to_csv(OUTPUT_PATH, index=False)
print(f"✅ Wrote: {OUTPUT_PATH}")
print(f"✅ NOTA: {n_nota} | INTACT: {n_intact} | Seed: {SEED}")
