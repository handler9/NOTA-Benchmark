"""
Computes inter-rater agreement between GPT-5 judge and Claude judge.
Compares behavior labels and premature_closure decisions across all 15 files.
Saves to metrics/healthbench_judge_agreement.csv and prints a summary.
"""

import pandas as pd
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

GPT_DIR    = Path("results_labeled_healthbench")
CLAUDE_DIR = Path("results_labeled_healthbench_claude")

rows = []
for gpt_path in sorted(GPT_DIR.glob("*_hb_labeled.csv")):
    claude_path = CLAUDE_DIR / gpt_path.name
    if not claude_path.exists():
        print(f"⚠️  Missing Claude file: {claude_path.name}")
        continue

    gpt_df    = pd.read_csv(gpt_path)
    claude_df = pd.read_csv(claude_path)

    bc_gpt    = [c for c in gpt_df.columns    if c.endswith("_behavior")][0]
    bc_claude = [c for c in claude_df.columns if c.endswith("_behavior")][0]
    pc_gpt    = [c for c in gpt_df.columns    if c.endswith("_premature_closure")][0]
    pc_claude = [c for c in claude_df.columns if c.endswith("_premature_closure")][0]

    prefix = bc_gpt.replace("_behavior", "")
    parts  = prefix.split("_")
    model, condition = parts[0], parts[1]

    gpt_renamed    = gpt_df[["prompt_id", "subset", bc_gpt, pc_gpt]].rename(
        columns={bc_gpt: "gpt_behavior", pc_gpt: "gpt_pc"})
    claude_renamed = claude_df[["prompt_id", bc_claude, pc_claude]].rename(
        columns={bc_claude: "claude_behavior", pc_claude: "claude_pc"})

    merged = gpt_renamed.merge(claude_renamed, on="prompt_id")

    # Drop rows where either judge errored
    valid = merged[
        ~merged["gpt_behavior"].isin(["PARSE_ERROR", "SKIP_ERROR"]) &
        ~merged["claude_behavior"].isin(["PARSE_ERROR", "SKIP_ERROR"])
    ].copy()

    for _, row in valid.iterrows():
        rows.append({
            "model":           model,
            "condition":       condition,
            "prompt_id":       row["prompt_id"],
            "subset":          row["subset"],
            "gpt_behavior":    row["gpt_behavior"],
            "claude_behavior": row["claude_behavior"],
            "behavior_agrees": row["gpt_behavior"] == row["claude_behavior"],
            "gpt_pc":          row["gpt_pc"],
            "claude_pc":       row["claude_pc"],
            "pc_agrees":       row["gpt_pc"] == row["claude_pc"],
        })

df = pd.DataFrame(rows)

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Total paired rows: {len(df)}")
print()

behavior_agree = df["behavior_agrees"].mean()
pc_agree       = df["pc_agrees"].mean()
print(f"Overall behavior agreement:          {behavior_agree:.1%}")
print(f"Overall premature_closure agreement: {pc_agree:.1%}")
print()

# Cohen's Kappa
try:
    kappa_behavior = cohen_kappa_score(df["gpt_behavior"], df["claude_behavior"])
    print(f"Cohen's Kappa (behavior):            {kappa_behavior:.3f}")
except Exception as e:
    print(f"Kappa (behavior) error: {e}")

valid_pc = df[df["gpt_pc"].notna() & df["claude_pc"].notna()]
try:
    kappa_pc = cohen_kappa_score(valid_pc["gpt_pc"], valid_pc["claude_pc"])
    print(f"Cohen's Kappa (premature_closure):   {kappa_pc:.3f}")
except Exception as e:
    print(f"Kappa (PC) error: {e}")

print()

# Agreement by model
print("=== Behavior Agreement by Model ===")
print(df.groupby("model")["behavior_agrees"].mean().round(3).to_string())
print()

# Agreement by condition
print("=== Behavior Agreement by Condition ===")
print(df.groupby("condition")["behavior_agrees"].mean().round(3).to_string())
print()

# Disagreement breakdown
disagree = df[~df["behavior_agrees"]]
print(f"=== Disagreement Cases ({len(disagree)}) ===")
confusion = pd.crosstab(disagree["gpt_behavior"], disagree["claude_behavior"],
                        rownames=["GPT-5"], colnames=["Claude"])
print(confusion.to_string())
print()

# Cases where GPT=CLARIFIED but Claude disagrees (most relevant for our known issue)
gpt_clarified = df[(df["gpt_behavior"] == "CLARIFIED") & (~df["behavior_agrees"])]
print(f"Cases where GPT said CLARIFIED but Claude disagreed: {len(gpt_clarified)}")
print(df[df["gpt_behavior"] == "CLARIFIED"]["claude_behavior"].value_counts().to_string())

# Save full row-level
out_path = Path("metrics/healthbench_judge_agreement.csv")
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path} ({len(df)} rows)")
