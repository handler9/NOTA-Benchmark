"""
Produces an aggregate results table grouped by model × condition only (all 394 questions per cell).
Saved to metrics/healthbench_results_aggregate.csv
"""

import pandas as pd
from pathlib import Path

df = pd.read_csv("metrics/healthbench_rowlevel.csv")

df["pc_binary"]    = (df["premature_closure"] == "yes").astype(float)
df["over_deferral"] = (df["behavior"] == "CLARIFIED").astype(float)
df["score_pct"]    = df["score_earned"] / df["score_possible"].replace(0, float("nan"))

MODEL_LABEL     = {"gpt": "GPT-5", "claude": "Claude", "gemini": "Gemini", "llama": "Llama", "deepseek": "DeepSeek"}
CONDITION_LABEL = {"baseline": "Baseline", "safety": "Safety", "clarification": "Clarification-First"}

def pct(x):
    return f"{round(x * 100, 1)}%"

rows = []
for (model, condition), g in df.groupby(["model", "condition"]):
    rows.append({
        "Model":                    MODEL_LABEL.get(model, model),
        "Condition":                CONDITION_LABEL.get(condition, condition),
        "N":                        len(g),
        "Premature Closure Rate":   pct(g["pc_binary"].mean()),
        "% Answered":               pct(g["behavior"].eq("ANSWERED").mean()),
        "% Clarified":              pct(g["behavior"].eq("CLARIFIED").mean()),
        "% Escalated":              pct(g["behavior"].eq("ESCALATED").mean()),
        "% Partial":                pct(g["behavior"].eq("PARTIAL").mean()),
        "Over-Deferral Rate":       pct(g["over_deferral"].mean()),
        "Mean Rubric Score":        pct(g["score_pct"].mean()),
    })

out = pd.DataFrame(rows)

model_order     = ["GPT-5", "Claude", "Gemini", "Llama", "DeepSeek"]
condition_order = ["Baseline", "Safety", "Clarification-First"]

out["Model"]     = pd.Categorical(out["Model"],     categories=model_order,     ordered=True)
out["Condition"] = pd.Categorical(out["Condition"], categories=condition_order, ordered=True)
out = out.sort_values(["Model", "Condition"]).reset_index(drop=True)

out.to_csv("metrics/healthbench_results_aggregate.csv", index=False)
print(f"Saved → metrics/healthbench_results_aggregate.csv ({len(out)} rows × {len(out.columns)} columns)")
print(out.to_string(index=False))
