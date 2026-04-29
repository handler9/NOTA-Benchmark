"""
HealthBench Analysis Script
Reads all labeled files from results_labeled_healthbench/ and computes:
  - Premature closure rate by model, condition, subset
  - Behavior label distribution
  - Rubric score (earned / possible) by model, condition, subset
  - Over-deferral rate on sufficient_context subset
Saves summary tables to metrics/healthbench_*.csv
"""

import pandas as pd
import glob
import json
from pathlib import Path

INPUT_DIR  = Path("results_labeled_healthbench")
OUTPUT_DIR = Path("metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------
# 1. Load all labeled files
# ------------------------------------------------------
rows = []
for f in sorted(glob.glob(str(INPUT_DIR / "*.csv"))):
    df = pd.read_csv(f)

    behavior_cols = [c for c in df.columns if c.endswith("_behavior")]
    pc_cols       = [c for c in df.columns if c.endswith("_premature_closure")]
    score_e_cols  = [c for c in df.columns if c.endswith("_score_earned")]
    score_p_cols  = [c for c in df.columns if c.endswith("_score_possible")]
    reasoning_cols= [c for c in df.columns if c.endswith("_reasoning")]

    if not behavior_cols:
        continue

    prefix    = behavior_cols[0].replace("_behavior", "")
    parts     = prefix.split("_")
    model     = parts[0]
    condition = parts[1]

    df["model"]     = model
    df["condition"] = condition
    df["behavior"]  = df[behavior_cols[0]]
    df["premature_closure"] = df[pc_cols[0]]    if pc_cols    else None
    df["score_earned"]      = df[score_e_cols[0]] if score_e_cols else None
    df["score_possible"]    = df[score_p_cols[0]] if score_p_cols else None
    df["reasoning"]         = df[reasoning_cols[0]] if reasoning_cols else None

    rows.append(df[[
        "prompt_id", "question", "subset", "example_tags",
        "model", "condition", "behavior", "premature_closure",
        "score_earned", "score_possible", "reasoning"
    ]])

all_df = pd.concat(rows, ignore_index=True)

# Drop parse errors and skip errors for main metrics
valid = all_df[~all_df["behavior"].isin(["PARSE_ERROR", "SKIP_ERROR"])].copy()
valid["pc_binary"] = (valid["premature_closure"] == "yes").astype(float)
valid["score_pct"]  = valid["score_earned"] / valid["score_possible"].replace(0, float("nan"))

print(f"Total labeled rows: {len(all_df)}")
print(f"Valid rows (excl. parse/skip errors): {len(valid)}")
print(f"Models: {sorted(valid['model'].unique())}")
print(f"Conditions: {sorted(valid['condition'].unique())}")
print(f"Subsets: {sorted(valid['subset'].unique())}")
print()

# ------------------------------------------------------
# 2. Premature closure rate — model × condition
# ------------------------------------------------------
pc_model = (
    valid.groupby(["model", "condition"])["pc_binary"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "premature_closure_rate", "count": "n"})
    .round(3)
    .unstack("condition")
)
pc_model.columns = ["_".join(c) for c in pc_model.columns]
print("=== Premature Closure Rate by Model & Condition ===")
print(pc_model.to_string())
print()

# ------------------------------------------------------
# 3. Premature closure rate — subset × condition
# ------------------------------------------------------
pc_subset = (
    valid.groupby(["subset", "condition"])["pc_binary"]
    .mean().round(3).unstack("condition")
)
print("=== Premature Closure Rate by Subset & Condition ===")
print(pc_subset.to_string())
print()

# ------------------------------------------------------
# 4. Behavior distribution — condition × behavior
# ------------------------------------------------------
behavior_dist = (
    valid.groupby(["condition", "behavior"]).size()
    .unstack("behavior", fill_value=0)
)
print("=== Behavior Distribution by Condition ===")
print(behavior_dist.to_string())
print()

# ------------------------------------------------------
# 5. Over-deferral on sufficient_context subset
# ------------------------------------------------------
sc = valid[valid["subset"] == "sufficient_context"].copy()
sc["over_deferral"] = (sc["behavior"] == "CLARIFIED").astype(float)
od = sc.groupby(["model", "condition"])["over_deferral"].mean().round(3).unstack("condition")
print("=== Over-Deferral Rate (sufficient_context subset) ===")
print(od.to_string())
print()

# ------------------------------------------------------
# 6. Rubric score — model × condition
# ------------------------------------------------------
score_model = (
    valid.groupby(["model", "condition"])["score_pct"]
    .mean().round(3).unstack("condition")
)
print("=== Mean Rubric Score % by Model & Condition ===")
print(score_model.to_string())
print()

# ------------------------------------------------------
# 7. Rubric score — subset × condition
# ------------------------------------------------------
score_subset = (
    valid.groupby(["subset", "condition"])["score_pct"]
    .mean().round(3).unstack("condition")
)
print("=== Mean Rubric Score % by Subset & Condition ===")
print(score_subset.to_string())
print()

# ------------------------------------------------------
# 8. Full row-level output
# ------------------------------------------------------
rowlevel_path = OUTPUT_DIR / "healthbench_rowlevel.csv"
valid.to_csv(rowlevel_path, index=False)
print(f"Saved row-level data → {rowlevel_path}")

# ------------------------------------------------------
# 9. Save summary tables
# ------------------------------------------------------
# Premature closure summary
pc_full = (
    valid.groupby(["model", "condition", "subset"])["pc_binary"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "premature_closure_rate", "count": "n"})
    .reset_index()
    .round(3)
)
pc_full.to_csv(OUTPUT_DIR / "healthbench_premature_closure.csv", index=False)
print(f"Saved premature closure summary → metrics/healthbench_premature_closure.csv")

# Behavior distribution
behavior_full = (
    valid.groupby(["model", "condition", "subset", "behavior"])
    .size().reset_index(name="n")
)
behavior_full.to_csv(OUTPUT_DIR / "healthbench_behavior_distribution.csv", index=False)
print(f"Saved behavior distribution → metrics/healthbench_behavior_distribution.csv")

# Rubric scores
score_full = (
    valid.groupby(["model", "condition", "subset"])["score_pct"]
    .agg(["mean", "std", "count"])
    .rename(columns={"mean": "mean_score_pct", "std": "std_score_pct", "count": "n"})
    .reset_index()
    .round(3)
)
score_full.to_csv(OUTPUT_DIR / "healthbench_rubric_scores.csv", index=False)
print(f"Saved rubric scores → metrics/healthbench_rubric_scores.csv")

# Over-deferral
od_full = (
    sc.groupby(["model", "condition"])["over_deferral"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "over_deferral_rate", "count": "n"})
    .reset_index()
    .round(3)
)
od_full.to_csv(OUTPUT_DIR / "healthbench_over_deferral.csv", index=False)
print(f"Saved over-deferral → metrics/healthbench_over_deferral.csv")

print("\nDone.")
