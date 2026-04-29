"""
Produces a single wide-format summary CSV with all key metrics per model × condition × subset.
Saved to metrics/healthbench_summary.csv
"""

import pandas as pd
from pathlib import Path

INPUT  = Path("metrics/healthbench_rowlevel.csv")
OUTPUT = Path("metrics/healthbench_summary.csv")

df = pd.read_csv(INPUT)

# Premature closure
df["pc_binary"] = (df["premature_closure"] == "yes").astype(float)

# Over-deferral (only meaningful for sufficient_context but compute for all)
df["over_deferral"] = (df["behavior"] == "CLARIFIED").astype(float)

# Rubric score %
df["score_pct"] = df["score_earned"] / df["score_possible"].replace(0, float("nan"))

# Behavior flags
df["is_answered"]   = (df["behavior"] == "ANSWERED").astype(float)
df["is_clarified"]  = (df["behavior"] == "CLARIFIED").astype(float)
df["is_escalated"]  = (df["behavior"] == "ESCALATED").astype(float)
df["is_partial"]    = (df["behavior"] == "PARTIAL").astype(float)

summary = (
    df.groupby(["model", "condition", "subset"])
    .agg(
        n                    = ("pc_binary",     "count"),
        premature_closure_rate = ("pc_binary",   "mean"),
        over_deferral_rate   = ("over_deferral", "mean"),
        pct_answered         = ("is_answered",   "mean"),
        pct_clarified        = ("is_clarified",  "mean"),
        pct_escalated        = ("is_escalated",  "mean"),
        pct_partial          = ("is_partial",    "mean"),
        mean_rubric_score_pct= ("score_pct",     "mean"),
        std_rubric_score_pct = ("score_pct",     "std"),
    )
    .round(3)
    .reset_index()
)

summary.to_csv(OUTPUT, index=False)
print(f"Saved → {OUTPUT}")
print(f"{len(summary)} rows × {len(summary.columns)} columns")
print(summary.to_string(index=False))
