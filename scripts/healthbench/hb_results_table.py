"""
Produces a single clean, human-readable CSV summarizing the key results.
Saved to metrics/healthbench_results_table.csv
"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("metrics/healthbench_rowlevel.csv")

df["pc_binary"] = (df["premature_closure"] == "yes").astype(float)
df["over_deferral"] = (df["behavior"] == "CLARIFIED").astype(float)
df["score_pct"] = df["score_earned"] / df["score_possible"].replace(0, float("nan"))

CONDITION_LABEL = {"baseline": "Baseline", "safety": "Safety", "clarification": "Clarification-First"}
MODEL_LABEL = {"gpt": "GPT-5", "claude": "Claude", "gemini": "Gemini", "llama": "Llama", "deepseek": "DeepSeek"}
SUBSET_LABEL = {
    "underspecified_low_risk":       "Underspecified (Low-Risk)",
    "underspecified_urgent":         "Underspecified (Urgent)",
    "sufficient_context":            "Sufficient Context (Control)",
    "hedging_reducible_uncertainty": "Hedging / Uncertainty",
}

rows = []
for (model, condition, subset), g in df.groupby(["model", "condition", "subset"]):
    sc_group = df[(df["model"] == model) & (df["condition"] == condition) & (df["subset"] == "sufficient_context")]
    over_def = sc_group["over_deferral"].mean() if subset == "sufficient_context" else None

    def pct(x):
        return f"{round(x * 100, 1)}%"

    rows.append({
        "Model":                    MODEL_LABEL.get(model, model),
        "Condition":                CONDITION_LABEL.get(condition, condition),
        "Subset":                   SUBSET_LABEL.get(subset, subset),
        "N":                        len(g),
        "Premature Closure Rate":   pct(g["pc_binary"].mean()),
        "% Answered":               pct(g["behavior"].eq("ANSWERED").mean()),
        "% Clarified":              pct(g["behavior"].eq("CLARIFIED").mean()),
        "% Escalated":              pct(g["behavior"].eq("ESCALATED").mean()),
        "% Partial":                pct(g["behavior"].eq("PARTIAL").mean()),
        "Over-Deferral Rate (Control)": pct(over_def) if over_def is not None else "",
        "Mean Rubric Score":        pct(g["score_pct"].mean()),
    })

out = pd.DataFrame(rows)

# Sort for readability
model_order     = ["GPT-5", "Claude", "Gemini", "Llama", "DeepSeek"]
condition_order = ["Baseline", "Safety", "Clarification-First"]
subset_order    = ["Underspecified (Low-Risk)", "Underspecified (Urgent)",
                   "Sufficient Context (Control)", "Hedging / Uncertainty"]

out["Model"]     = pd.Categorical(out["Model"],     categories=model_order,     ordered=True)
out["Condition"] = pd.Categorical(out["Condition"], categories=condition_order, ordered=True)
out["Subset"]    = pd.Categorical(out["Subset"],    categories=subset_order,    ordered=True)
out = out.sort_values(["Model", "Condition", "Subset"]).reset_index(drop=True)

out.to_csv("metrics/healthbench_results_pct.csv", index=False)
print(f"Saved → metrics/healthbench_results_pct.csv ({len(out)} rows × {len(out.columns)} columns)")
