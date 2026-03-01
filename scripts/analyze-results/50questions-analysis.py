#!/usr/bin/env python

import os
import glob
import pandas as pd

# --------------------------------------------------------
# 0. Paths & config
# --------------------------------------------------------

DATA_DIR = "data"
RESULTS_DIR = "results_raw_50q_test"
METRICS_DIR = "metrics"

# Try data/50question_key.csv first, then ./50question_key.csv
QUESTION_KEY_CANDIDATES = [
    os.path.join(DATA_DIR, "50question_key.csv"),
    "50question_key.csv",
]

QUESTION_KEY_FILE = None
for p in QUESTION_KEY_CANDIDATES:
    if os.path.exists(p):
        QUESTION_KEY_FILE = p
        break

if QUESTION_KEY_FILE is None:
    raise FileNotFoundError(
        "Could not find 50question_key.csv in data/ or repo root."
    )

print(f"✅ Using key: {QUESTION_KEY_FILE}")

# Look at all CSVs in results_raw_judgetest/
RESULTS_GLOB = os.path.join(RESULTS_DIR, "*.csv")
print(f"🔍 Looking for result files matching: {RESULTS_GLOB}")
result_files = glob.glob(RESULTS_GLOB)
if not result_files:
    raise FileNotFoundError(
        f"No result CSVs found matching pattern: {RESULTS_GLOB}"
    )
print("   Found:")
for f in result_files:
    print("   -", f)

# --------------------------------------------------------
# 1. Load question key
# --------------------------------------------------------

key = pd.read_csv(QUESTION_KEY_FILE)

if "question_id" not in key.columns or "correct_choice" not in key.columns:
    raise ValueError(
        "Key file must have 'question_id' and 'correct_choice' columns."
    )

key["correct_choice_norm"] = (
    key["correct_choice"]
    .astype(str)
    .str.strip()
    .str.upper()
)

key = key[["question_id", "correct_choice_norm"]]

os.makedirs(METRICS_DIR, exist_ok=True)

summary_rows = []

# --------------------------------------------------------
# 2. Loop over all result files
# --------------------------------------------------------

for path in sorted(result_files):
    print(f"\n📄 Processing {path}")
    df = pd.read_csv(path)
    print("   Columns:", list(df.columns))

    # ---- detect the choice column (e.g. gpt5_choice, claude_choice, etc.) ----
    choice_cols = [
        c for c in df.columns
        if c.endswith("_choice") and c != "correct_choice"
    ]
    if len(choice_cols) != 1:
        raise ValueError(
            f"Expected exactly one '*_choice' column in {path}, "
            f"got {choice_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    choice_col = choice_cols[0]
    print(f"   Using choice column: {choice_col}")

    # ---- detect abstain column (e.g. gpt5_abstain_code) if present ----
    abstain_cols = [c for c in df.columns if "abstain" in c.lower()]
    abstain_col = abstain_cols[0] if abstain_cols else None
    if abstain_col:
        print(f"   Using abstain column: {abstain_col}")
    else:
        print("   No abstain column found; will infer from choice text only.")

    # Normalize model choice (A/B/C/D/E/NOTA/etc.)
    df["model_choice_norm"] = (
        df[choice_col]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Flag abstains
    if abstain_col is not None:
        ac = df[abstain_col].astype(str).str.upper()
        df["is_abstain"] = ac.isin(
            [
                "1",
                "TRUE",
                "T",
                "YES",
                "Y",
                "NO_VALID_OPTION",
                "NO VALID OPTION",
                "ABSTAIN",
                "NOTA",
                "NONE",
                "NA",
            ]
        )
    else:
        df["is_abstain"] = df["model_choice_norm"].isin(
            ["NOTA", "NONE", "NO VALID OPTION", "NO_VALID_OPTION", "ABSTAIN"]
        )

    # Merge with key on question_id
    if "question_id" not in df.columns:
        raise ValueError(f"'question_id' column missing in {path}")

    merged = pd.merge(
        df,
        key,
        on="question_id",
        how="inner",
    )

    # Correct if matches AND not abstain
    merged["is_correct"] = (
        (merged["model_choice_norm"] == merged["correct_choice_norm"])
        & (~merged["is_abstain"])
    )

    # Basic counts
    n_total = len(merged)
    n_abstain = int(merged["is_abstain"].sum())
    n_answered = n_total - n_abstain
    n_correct = int(merged["is_correct"].sum())

    accuracy_all = n_correct / n_total if n_total else 0.0
    accuracy_answered = n_correct / n_answered if n_answered > 0 else 0.0
    abstain_rate = n_abstain / n_total if n_total else 0.0

    fname = os.path.basename(path)

    # crude model name parser: take first chunk before "_"
    model = fname.split("_")[0]

    summary_rows.append(
        {
            "file": fname,
            "model": model,
            "n_total": n_total,
            "n_correct": n_correct,
            "n_abstain": n_abstain,
            "n_answered": n_answered,
            "accuracy_overall": accuracy_all,
            "accuracy_answered_only": accuracy_answered,
            "abstain_rate": abstain_rate,
        }
    )

    # Save per-question breakdown
    per_q_out = os.path.join(METRICS_DIR, f"per_question_50intact_{fname}")
    merged.to_csv(per_q_out, index=False)
    print(f"   → Saved per-question results to {per_q_out}")
    print(
        f"   → accuracy_overall = {accuracy_all:.3f}, "
        f"accuracy_answered_only = {accuracy_answered:.3f}, "
        f"abstain_rate = {abstain_rate:.3f}"
    )

# --------------------------------------------------------
# 3. Save summary table
# --------------------------------------------------------

summary = pd.DataFrame(summary_rows)
summary_out = os.path.join(METRICS_DIR, "summary_50intact6.csv")
summary.to_csv(summary_out, index=False)

print(f"\n✅ Summary saved to: {summary_out}")
print(summary.to_string(index=False))
