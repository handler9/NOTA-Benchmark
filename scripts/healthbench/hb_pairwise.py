"""
Pairwise item-level analysis for HealthBench.

For each question, holds the question constant and tests whether the model
changed behavior or premature closure status when the prompt condition changed.

Tests:
  - Baseline → Safety
  - Baseline → Clarification-First
  - Baseline → Refined-Clarification (if available)

Per model:
  - % questions where behavior changed
  - % questions where premature closure changed (yes→no or no→yes)
  - McNemar test on pc_binary pairs (improvement = yes→no)

Saves to metrics/healthbench_pairwise.csv and prints a summary.

Run from project root:
  python scripts/healthbench/hb_pairwise.py
  python scripts/healthbench/hb_pairwise.py --judge gpt   # use GPT-5 v2 judge instead
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.stats.contingency_tables import mcnemar

USE_GPT_JUDGE = "--judge" in sys.argv and sys.argv[sys.argv.index("--judge") + 1] == "gpt"
LABELED_DIR = Path("results_labeled_healthbench_v2") if USE_GPT_JUDGE else Path("results_labeled_healthbench_claude")
JUDGE_NAME  = "gpt5v2" if USE_GPT_JUDGE else "claude"
OUT_PATH    = Path(f"metrics/healthbench_pairwise_{JUDGE_NAME}.csv")

CONDITION_MAP = {
    "baseline":      "Baseline",
    "safety":        "Safety",
    "clarification": "Clarification-First",
    "refined":       "Refined-Clarification",
}

MODEL_LABEL_MAP = {
    "gpt":      "GPT-5",
    "claude":   "Claude",
    "gemini":   "Gemini",
    "llama":    "Llama",
    "deepseek": "DeepSeek",
}

# ── Load all labeled files ────────────────────────────────────────────────────
records = []
for path in sorted(LABELED_DIR.glob("*_hb_labeled.csv")):
    df = pd.read_csv(path)
    bc = [c for c in df.columns if c.endswith("_behavior")][0]
    pc = [c for c in df.columns if c.endswith("_premature_closure")][0]
    prefix = bc.replace("_behavior", "")
    parts  = prefix.split("_")
    model_key = parts[0]
    condition_key = parts[1]

    # Drop errors
    df = df[~df[bc].isin(["PARSE_ERROR", "SKIP_ERROR"])].copy()

    for _, row in df.iterrows():
        records.append({
            "model":     model_key,
            "condition": condition_key,
            "prompt_id": row["prompt_id"],
            "subset":    row.get("subset", ""),
            "behavior":  row[bc],
            "pc":        row[pc],
        })

all_df = pd.DataFrame(records)
all_df["pc_binary"] = (all_df["pc"] == "yes").astype(int)

print(f"Loaded {len(all_df)} rows from {LABELED_DIR}/")
print(f"Models:     {sorted(all_df['model'].unique())}")
print(f"Conditions: {sorted(all_df['condition'].unique())}")
print()

# ── Pairwise comparison function ─────────────────────────────────────────────
def pairwise_compare(model_df, cond_a, cond_b):
    """
    For a single model, compare two conditions at the question level.
    Returns summary stats and McNemar test result on pc_binary.
    """
    a = model_df[model_df["condition"] == cond_a][["prompt_id", "behavior", "pc_binary"]].rename(
        columns={"behavior": "behavior_a", "pc_binary": "pc_a"})
    b = model_df[model_df["condition"] == cond_b][["prompt_id", "behavior", "pc_binary"]].rename(
        columns={"behavior": "behavior_b", "pc_binary": "pc_b"})

    merged = a.merge(b, on="prompt_id")
    if len(merged) == 0:
        return None

    n = len(merged)

    # Behavior change rate
    behavior_changed = (merged["behavior_a"] != merged["behavior_b"]).mean()

    # PC change rates
    pc_improved   = ((merged["pc_a"] == 1) & (merged["pc_b"] == 0)).sum()  # yes→no (good)
    pc_worsened   = ((merged["pc_a"] == 0) & (merged["pc_b"] == 1)).sum()  # no→yes (bad)
    pc_unchanged  = ((merged["pc_a"] == merged["pc_b"])).sum()
    pc_change_rate = (merged["pc_a"] != merged["pc_b"]).mean()

    # McNemar test on pc_binary (2x2 table)
    # Table: [[both_no, a_no_b_yes], [a_yes_b_no, both_yes]]
    both_no    = ((merged["pc_a"] == 0) & (merged["pc_b"] == 0)).sum()
    both_yes   = ((merged["pc_a"] == 1) & (merged["pc_b"] == 1)).sum()
    a_yes_b_no = pc_improved
    a_no_b_yes = pc_worsened

    table = np.array([[both_no, a_no_b_yes],
                      [a_yes_b_no, both_yes]])
    try:
        result = mcnemar(table, exact=False, correction=True)
        p_mcnemar = result.pvalue
    except Exception:
        p_mcnemar = None

    return {
        "n_paired":           n,
        "behavior_change_rate": round(behavior_changed, 4),
        "pc_improved":        int(pc_improved),    # yes→no
        "pc_worsened":        int(pc_worsened),    # no→yes
        "pc_unchanged":       int(pc_unchanged),
        "pc_change_rate":     round(pc_change_rate, 4),
        "pc_rate_a":          round(merged["pc_a"].mean(), 4),
        "pc_rate_b":          round(merged["pc_b"].mean(), 4),
        "p_mcnemar":          round(p_mcnemar, 4) if p_mcnemar is not None else None,
    }


# ── Run all comparisons ───────────────────────────────────────────────────────
COMPARISONS = [
    ("baseline", "safety"),
    ("baseline", "clarification"),
    ("baseline", "refined"),
    ("clarification", "refined"),
]

rows_out = []
for model_key in sorted(all_df["model"].unique()):
    model_df = all_df[all_df["model"] == model_key]
    available_conditions = model_df["condition"].unique()

    for cond_a, cond_b in COMPARISONS:
        if cond_a not in available_conditions or cond_b not in available_conditions:
            continue
        result = pairwise_compare(model_df, cond_a, cond_b)
        if result is None:
            continue
        rows_out.append({
            "Model":       MODEL_LABEL_MAP.get(model_key, model_key),
            "Condition A": CONDITION_MAP.get(cond_a, cond_a),
            "Condition B": CONDITION_MAP.get(cond_b, cond_b),
            **result,
        })

results_df = pd.DataFrame(rows_out)

# ── Print summary ─────────────────────────────────────────────────────────────
print("=" * 80)
print(f"PAIRWISE ITEM-LEVEL ANALYSIS  |  Judge: {JUDGE_NAME.upper()}")
print("=" * 80)

for _, row in results_df.iterrows():
    sig = "***" if (row["p_mcnemar"] is not None and row["p_mcnemar"] < 0.001) else \
          "**"  if (row["p_mcnemar"] is not None and row["p_mcnemar"] < 0.01)  else \
          "*"   if (row["p_mcnemar"] is not None and row["p_mcnemar"] < 0.05)  else "ns"
    print(
        f"{row['Model']:10s}  {row['Condition A']:22s} → {row['Condition B']:25s}  "
        f"n={row['n_paired']}  "
        f"behavior_changed={row['behavior_change_rate']:.1%}  "
        f"PC: {row['pc_rate_a']:.1%}→{row['pc_rate_b']:.1%}  "
        f"improved={row['pc_improved']} worsened={row['pc_worsened']}  "
        f"p={row['p_mcnemar']} {sig}"
    )

print()
results_df.to_csv(OUT_PATH, index=False)
print(f"Saved → {OUT_PATH}")
