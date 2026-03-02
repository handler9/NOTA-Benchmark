import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Optional

# =========================
# CONFIG
# =========================
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "llm_runs"
QUESTION_COL = "question_id"

# Matches: model_prompt_run.csv
FILENAME_REGEX = re.compile(
    r"^(?P<model>[^_]+)_(?P<prompt>[^_]+)_(?P<run>\d+)\.csv$",
    re.I
)

ABSTAIN_TOKENS = {"NO_VALID_OPTION", "ABSTAIN"}

# =========================
# HELPERS
# =========================
def shannon_entropy(values: pd.Series) -> float:
    counts = values.value_counts(dropna=False).values
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(-(probs * np.log2(probs)).sum())

def find_answer_column(columns) -> Optional[str]:
    cols = list(columns)

    # Prefer "*_choice" columns (claude_choice, gpt_choice, etc.)
    choice_cols = [c for c in cols if c.lower().endswith("_choice")]
    if choice_cols:
        return choice_cols[0]

    # Common fallbacks
    for c in ["model_answer", "answer", "final_answer", "choice", "prediction"]:
        if c in cols:
            return c

    return None

# =========================
# LOAD FILES
# =========================
rows = []
skipped = []
seen_keys = set()

for fp in sorted(DATA_DIR.glob("*.csv")):
    m = FILENAME_REGEX.search(fp.name)
    if not m:
        skipped.append(fp.name)
        continue

    meta = m.groupdict()
    model = meta["model"]
    prompt = meta["prompt"]
    run = int(meta["run"])

    key = (model, prompt, run)
    if key in seen_keys:
        raise ValueError(f"Duplicate model/prompt/run detected: {key} (file: {fp.name})")
    seen_keys.add(key)

    df = pd.read_csv(fp)

    if QUESTION_COL not in df.columns:
        raise ValueError(f"{fp.name} missing required '{QUESTION_COL}' column")

    ans_col = find_answer_column(df.columns)
    if ans_col is None:
        raise ValueError(f"{fp.name} has no recognizable answer column. Columns: {list(df.columns)}")

    df = df[[QUESTION_COL, ans_col]].copy()
    df = df.rename(columns={ans_col: "model_answer"})

    df["model"] = model
    df["prompt"] = prompt
    df["run"] = run
    df["source_file"] = fp.name

    rows.append(df)

if not rows:
    raise RuntimeError(f"No CSVs loaded from {DATA_DIR.resolve()}. Check folder path and filename pattern.")

all_df = pd.concat(rows, ignore_index=True)

print(f"Loaded {len(rows)} files, {len(all_df)} total rows.")
if skipped:
    print(f"Skipped {len(skipped)} files (didn't match pattern). Example(s):")
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
# RUN COUNTS (GROUND TRUTH)
# =========================
runs_by_mp = (
    all_df.groupby(["model", "prompt"])["run"]
    .nunique()
    .reset_index(name="runs_observed")
)

# =========================
# PER-QUESTION VARIANCE
# =========================
group_cols = ["model", "prompt", QUESTION_COL]
gq = all_df.groupby(group_cols)

per_question = gq.agg(
    n_runs=("run", "nunique"),  # distinct runs per question
    n_unique=("model_answer", lambda x: x.nunique(dropna=False)),
    stable=("model_answer", lambda x: x.nunique(dropna=False) == 1),
    has_abstain=("model_answer", lambda x: x.isin(ABSTAIN_TOKENS).any()),
    all_abstain=("model_answer", lambda x: x.isin(ABSTAIN_TOKENS).all()),
    entropy=("model_answer", shannon_entropy),
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
        runs_min_per_question=("n_runs", "min"),
        runs_median_per_question=("n_runs", "median"),
        runs_max_per_question=("n_runs", "max"),
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
    .merge(runs_by_mp, on=["model", "prompt"], how="left")
)

summary["coverage_min_pct"] = (summary["runs_min_per_question"] / summary["runs_observed"] * 100).round(2)
summary["coverage_median_pct"] = (summary["runs_median_per_question"] / summary["runs_observed"] * 100).round(2)

summary["stable_pct"] = (summary["stable_pct"] * 100).round(2)
summary["unstable_pct"] = (summary["unstable_pct"] * 100).round(2)
summary["abstain_flip_pct"] = (summary["abstain_flip_pct"] * 100).round(2)
summary["entropy_mean"] = summary["entropy_mean"].round(4)
summary["entropy_median"] = summary["entropy_median"].round(4)

summary = summary.sort_values(["prompt", "model"])
summary.to_csv(REPO_ROOT / "metrics" / "variance_summary_by_model_prompt.csv", index=False)

# =========================
# QC WARNINGS
# =========================
missing = summary[summary["coverage_min_pct"] < 100]
if not missing.empty:
    print("\nWARNING: Some model×prompt combos have missing questions in one or more runs:")
    print(missing[["model", "prompt", "runs_observed", "runs_min_per_question", "coverage_min_pct"]].to_string(index=False))

print("\nSaved:")
print(" - manifest_files_loaded.csv")
print(" - variance_per_question.csv")
print(" - variance_summary_by_model_prompt.csv")
