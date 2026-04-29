"""
Builds a stratified audit sample for manual review of judge decisions.

Samples across four audit categories:
  A. Premature closure = yes  (judge said model answered when it shouldn't have)
  B. Over-deferral: CLARIFIED/ESCALATED on sufficient_context  (judge said model over-deferred)
  C. Control - correct ANSWERED on sufficient_context  (judge said model answered correctly)
  D. Control - correct CLARIFIED/ESCALATED on underspecified  (judge said model correctly held back)

Each category: up to 10 cases sampled across models/conditions.
Output includes the question, model response, judge behavior, premature_closure, and reasoning.

Saved to metrics/healthbench_judge_audit.csv
"""

import pandas as pd
import glob
import json
import random
from pathlib import Path
from langdetect import detect, LangDetectException

random.seed(42)

def is_english(text):
    try:
        return detect(str(text)) == "en"
    except LangDetectException:
        return False

import sys
USE_CLAUDE_JUDGE = "--claude" in sys.argv

LABELED_DIR  = Path("results_labeled_healthbench_claude") if USE_CLAUDE_JUDGE else Path("results_labeled_healthbench")
RAW_DIR      = Path("results_raw_healthbench")
SOURCE_JSONL = Path("data/healthbench/healthbench_subset_combined.jsonl")
JUDGE_NAME   = "claude" if USE_CLAUDE_JUDGE else "gpt5"
OUT_PATH     = Path(f"metrics/healthbench_judge_audit_{JUDGE_NAME}.csv")

# ── Load full conversation threads from source JSONL ─────────────────────────
def format_conversation(prompt_turns):
    """Format a list of {role, content} turns into a readable string."""
    lines = []
    for turn in prompt_turns:
        role = "Patient" if turn["role"] == "user" else "Assistant (prior)"
        lines.append(f"[{role}]: {turn['content']}")
    return "\n".join(lines)

full_conversation_map = {}
with open(SOURCE_JSONL) as f:
    for line in f:
        row = json.loads(line)
        turns = row.get("prompt", [])
        full_conversation_map[row["prompt_id"]] = format_conversation(turns)

# ── Load all labeled files with raw response text ────────────────────────────
rows = []
for labeled_path in sorted(LABELED_DIR.glob("*_hb_labeled.csv")):
    df = pd.read_csv(labeled_path)

    bc   = [c for c in df.columns if c.endswith("_behavior")][0]
    pc   = [c for c in df.columns if c.endswith("_premature_closure")][0]
    rc   = [c for c in df.columns if c.endswith("_reasoning")]
    prefix = bc.replace("_behavior", "")
    parts  = prefix.split("_")
    model, condition = parts[0], parts[1]

    # Load raw response
    raw_path = RAW_DIR / labeled_path.name.replace("_labeled", "").replace("_hb_labeled", "_hb")
    if not raw_path.exists():
        continue
    raw_df  = pd.read_csv(raw_path)
    raw_col = [c for c in raw_df.columns if c.endswith("_raw")][0]
    raw_map = dict(zip(raw_df["prompt_id"], raw_df[raw_col]))

    for _, row in df.iterrows():
        pid = row["prompt_id"]
        rows.append({
            "model":                    model,
            "condition":                condition,
            "prompt_id":                pid,
            "subset":                   row["subset"],
            "full_conversation":        full_conversation_map.get(pid, row["question"]),
            "question":                 row["question"],
            "model_response":           raw_map.get(pid, ""),
            "judge_behavior":           row[bc],
            "judge_premature_closure":  row[pc],
            "judge_reasoning":          row[rc[0]] if rc else "",
        })

all_df = pd.DataFrame(rows)

# Drop parse/skip errors
all_df = all_df[~all_df["judge_behavior"].isin(["PARSE_ERROR", "SKIP_ERROR"])].copy()

# Filter to English-only questions (use last turn / question field for detection)
all_df = all_df[all_df["question"].apply(is_english)].copy()

# Explicit over-deferral flag: CLARIFIED or ESCALATED on sufficient_context
all_df["judge_over_deferral"] = (
    (all_df["subset"] == "sufficient_context") &
    (all_df["judge_behavior"] == "CLARIFIED")
).map({True: "yes", False: "no"})

# ── Define audit categories ──────────────────────────────────────────────────

# A: Judge said premature closure — model answered when it shouldn't have
cat_a = all_df[
    (all_df["judge_premature_closure"] == "yes") &
    (all_df["subset"] != "sufficient_context")
].copy()
cat_a["audit_category"] = "A: Premature Closure (judge=yes)"

# B: Over-deferral — model clarified on sufficient_context (ESCALATED excluded: may be correct for emergencies)
cat_b = all_df[
    (all_df["subset"] == "sufficient_context") &
    (all_df["judge_behavior"] == "CLARIFIED")
].copy()
cat_b["audit_category"] = "B: Over-Deferral (clarified on sufficient_context)"

# C: Control correct — model answered directly on sufficient_context
cat_c = all_df[
    (all_df["subset"] == "sufficient_context") &
    (all_df["judge_behavior"] == "ANSWERED") &
    (all_df["judge_premature_closure"] == "no")
].copy()
cat_c["audit_category"] = "C: Correct Answer (answered on sufficient_context)"

# D: Control correct — model clarified/escalated on underspecified
cat_d = all_df[
    (all_df["subset"].isin(["underspecified_low_risk", "underspecified_urgent"])) &
    (all_df["judge_behavior"].isin(["CLARIFIED", "ESCALATED"])) &
    (all_df["judge_premature_closure"] == "no")
].copy()
cat_d["audit_category"] = "D: Correct Deferral (clarified or escalated on underspecified)"

# ── Sample 10 per category, stratified across models ────────────────────────
def stratified_sample(df, n=10):
    sampled = []
    models = df["model"].unique()
    per_model = max(1, n // len(models))
    for model in models:
        sub = df[df["model"] == model]
        k = min(per_model, len(sub))
        sampled.append(sub.sample(k, random_state=42))
    result = pd.concat(sampled).sample(frac=1, random_state=42)
    return result.head(n)

samples = pd.concat([
    stratified_sample(cat_a, 25),
    stratified_sample(cat_b, 25),
    stratified_sample(cat_c, 25),
    stratified_sample(cat_d, 25),
]).reset_index(drop=True)

# Clean up for readability — full_conversation replaces question as primary context
samples = samples[[
    "audit_category", "model", "condition", "subset",
    "full_conversation", "model_response",
    "judge_behavior", "judge_premature_closure", "judge_over_deferral", "judge_reasoning",
    "prompt_id"
]]

# Add blank auditor columns for manual review
samples["auditor_agrees"] = ""        # yes / no
samples["auditor_correct_label"] = "" # fill in if disagree
samples["auditor_notes"] = ""         # free text

samples.to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH} ({len(samples)} rows)")
print()
print("Audit category counts:")
print(samples["audit_category"].value_counts().to_string())
print()
print("Model distribution:")
print(samples["model"].value_counts().to_string())
