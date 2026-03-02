import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================
# CONFIG
# =========================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "llm_runs"

QUESTION_COL = "question_id"

FILENAME_REGEX = re.compile(
    r"^(?P<model>[^_]+)_(?P<prompt>[^_]+)_(?P<run>\d+)(?:csv)?\.csv$",
    re.I
)

ABSTAIN_TOKENS = {"NO_VALID_OPTION", "ABSTAIN"}

def shannon_entropy(values):
    counts = values.value_counts(dropna=False).values
    total = counts.sum()
    probs = counts / total
    return float(-(probs * np.log2(probs)).sum())

def find_answer_column(columns):
    # Prefer *_choice columns (claude_choice, gpt_choice, etc.)
    choice_cols = [c for c in columns if c.lower().endswith("_choice")]
    if choice_cols:
        return choice_cols[0]

    # Fallbacks
    for c in ["model_answer", "answer"]:
        if c in columns:
            return c

    return None

# =========================
# LOAD FILES
# =========================

rows = []
skipped = []

for fp in sorted(DATA_DIR.glob("*.csv")):
    m = FILENAME_REGEX.search(fp.name)
    if not m:
        skipped.append(fp.name)
        continue

    meta = m.groupdict()
    df = pd.read_csv(fp)

    if QUESTION_COL not in df.columns:
        raise ValueError(f"{fp.name} missing '{QUESTION_COL}' column")

    ans_col = find_answer_column(df.columns)
    if ans_col is None:
        raise ValueError(
            f"{fp.name} has no answer column. Found columns: {list(df.columns)}"
        )

    df = df[[QUESTION_COL, ans_col]].copy()
    df = df.rename(columns={ans_col: "model_answer"})

    df["model"] = meta["model"]
    df["prompt"] = meta["prompt"]
    df["run"] = int(meta["run"])
    df["source_file"] = fp.name
    rows.append(df)

all_df = pd.concat(rows, ignore_index=True)

print(f"Loaded {len(rows)} files, {len(all_df)} rows.")
if skipped:
    print(f"Skipped {len(skipped)} files (didn't match pattern):")
    for s in skipped[:10]:
        print(" -", s)

# =========================
# MANIFEST
# =========================

manifest = (
    all_df[["source_file", "model", "prompt", "run"]]
    .drop_duplicates()
    .sort_values(["prompt", "model", "run", "source_file"])
)
manifest.to_csv(REPO_ROOT / "data" / "manifest_files_loaded.csv", index=False)

# =========================
# PER-QUESTION VARIANCE
# =========================

group_cols = ["model", "prompt", QUESTION_COL]
g = all_df.groupby(group_cols)["model_answer"]

per_question = g.agg(
    n_runs="count",
    n_unique=lambda x: x.nunique(dropna=False),
    stable=lambda x: x.nunique(dropna=False) == 1,
    has_abstain=lambda x: x.isin(ABSTAIN_TOKENS).any(),
    all_abstain=lambda x: x.isin(ABSTAIN_TOKENS).all(),
    entropy=shannon_entropy,
).reset_index()

per_question["abstain_flip"] = per_question["has_abstain"] & (~per_question["all_abstain"])

per_question.to_csv(REPO_ROOT / "metrics" / "variance_per_question.csv", index=False)

# =========================
# SUMMARY BY MODEL × PROMPT
# =========================

summary = (
    per_question.groupby(["model", "prompt"])
    .agg(
        questions=(QUESTION_COL, "count"),
        runs_observed=("n_runs", "min"),
        stable_n=("stable", "sum"),
        stable_pct=("stable", "mean"),
        unstable_n=("stable", lambda s: (~s).sum()),
        unstable_pct=("stable", lambda s: (~s).mean()),
        entropy_mean=("entropy", "mean"),
        entropy_median=("entropy", "median"),
        abstain_flip_n=("abstain_flip", "sum"),
        abstain_flip_pct=("abstain_flip", "mean"),
    )
    .reset_index()
)

summary["stable_pct"] = (summary["stable_pct"] * 100).round(2)
summary["unstable_pct"] = (summary["unstable_pct"] * 100).round(2)
summary["abstain_flip_pct"] = (summary["abstain_flip_pct"] * 100).round(2)
summary["entropy_mean"] = summary["entropy_mean"].round(4)
summary["entropy_median"] = summary["entropy_median"].round(4)

summary = summary.sort_values(["prompt", "model"])
summary.to_csv(REPO_ROOT / "metrics" / "variance_summary_by_model_prompt.csv", index=False)

print("\nSaved:")
print(" - manifest_files_loaded.csv")
print(" - variance_per_question.csv")
print(" - variance_summary_by_model_prompt.csv")
