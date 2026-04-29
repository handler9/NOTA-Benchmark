"""
Generates all tables and figures for the HealthBench results section.

Tables:
  T1. Aggregate results — GPT-5 judge (all models × conditions)
  T2. Aggregate results — Claude judge (sensitivity analysis)
  T3. Inter-rater agreement summary

Figures:
  F1. Premature closure rate by model and condition (grouped bar)
  F2. Over-deferral rate by model and condition (grouped bar)
  F3. Mean rubric score by model and condition (grouped bar)
  F4. Premature closure rate by subset and condition (heatmap)
  F5. GPT-5 vs Claude judge: premature closure comparison (scatter)
  F6. Behavior distribution by condition (stacked bar)

Saved to metrics/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path("metrics/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
MODEL_ORDER     = ["GPT-5", "Claude", "Gemini", "Llama", "DeepSeek"]
CONDITION_ORDER = ["Baseline", "Safety", "Clarification-First"]
CONDITION_COLORS = {
    "Baseline":             "#4C72B0",
    "Safety":               "#DD8452",
    "Clarification-First":  "#55A868",
}
MODEL_COLORS = {
    "GPT-5":    "#4C72B0",
    "Claude":   "#DD8452",
    "Gemini":   "#55A868",
    "Llama":    "#C44E52",
    "DeepSeek": "#8172B3",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
gpt_agg    = pd.read_csv("metrics/healthbench_results_aggregate.csv")
claude_agg = pd.read_csv("metrics/healthbench_results_claudejudge.csv")
rowlevel   = pd.read_csv("metrics/healthbench_rowlevel.csv")
agreement  = pd.read_csv("metrics/healthbench_judge_agreement.csv")

# Parse pct strings to floats
def parse_pct(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].str.rstrip("%").astype(float) / 100
    return df

pct_cols = ["Premature Closure Rate", "Over-Deferral Rate", "Mean Rubric Score",
            "% Answered", "% Clarified", "% Escalated", "% Partial",
            "Premature Closure", "Over-Deferral"]
gpt_agg    = parse_pct(gpt_agg,    pct_cols)
claude_agg = parse_pct(claude_agg, pct_cols)


# ── Helper: grouped bar chart ─────────────────────────────────────────────────
def grouped_bar(ax, df, metric_col, ylabel, title, pct=True, ylim=None):
    x = np.arange(len(MODEL_ORDER))
    width = 0.25
    offsets = [-width, 0, width]

    for i, cond in enumerate(CONDITION_ORDER):
        vals = []
        for model in MODEL_ORDER:
            row = df[(df["Model"] == model) & (df["Condition"] == cond)]
            vals.append(row[metric_col].values[0] if len(row) else 0)
        bars = ax.bar(x + offsets[i], vals, width, label=cond,
                      color=CONDITION_COLORS[cond], alpha=0.88, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=10)
    if pct:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    if ylim:
        ax.set_ylim(ylim)
    ax.set_ylim(bottom=0)
    return ax


# ── F1: Premature Closure Rate ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
grouped_bar(ax, gpt_agg, "Premature Closure Rate",
            "Premature Closure Rate", "Figure 1. Premature Closure Rate by Model and Prompt Condition\n(GPT-5 Judge)", ylim=(0, 0.65))
fig.tight_layout()
fig.savefig(OUT_DIR / "F1_premature_closure.png", bbox_inches="tight")
plt.close()
print("Saved F1")

# ── F2: Over-Deferral Rate ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
grouped_bar(ax, gpt_agg, "Over-Deferral Rate",
            "Over-Deferral Rate", "Figure 2. Over-Deferral Rate by Model and Prompt Condition\n(GPT-5 Judge)", ylim=(0, 0.65))
fig.tight_layout()
fig.savefig(OUT_DIR / "F2_over_deferral.png", bbox_inches="tight")
plt.close()
print("Saved F2")

# ── F3: Mean Rubric Score ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
grouped_bar(ax, gpt_agg, "Mean Rubric Score",
            "Mean Rubric Score (%)", "Figure 3. Mean Rubric Score by Model and Prompt Condition\n(GPT-5 Judge)", ylim=(0, 1.0))
fig.tight_layout()
fig.savefig(OUT_DIR / "F3_rubric_score.png", bbox_inches="tight")
plt.close()
print("Saved F3")

# ── F1b / F2b: Claude judge versions ─────────────────────────────────────────
# Rename columns to match grouped_bar expectations
claude_agg_plot = claude_agg.rename(columns={
    "Premature Closure": "Premature Closure Rate",
    "Over-Deferral":     "Over-Deferral Rate",
})

fig, ax = plt.subplots(figsize=(10, 5))
grouped_bar(ax, claude_agg_plot, "Premature Closure Rate",
            "Premature Closure Rate", "Figure 1b. Premature Closure Rate by Model and Prompt Condition\n(Claude Judge)", ylim=(0, 0.65))
fig.tight_layout()
fig.savefig(OUT_DIR / "F1b_premature_closure_claudejudge.png", bbox_inches="tight")
plt.close()
print("Saved F1b")

fig, ax = plt.subplots(figsize=(10, 5))
grouped_bar(ax, claude_agg_plot, "Over-Deferral Rate",
            "Over-Deferral Rate", "Figure 2b. Over-Deferral Rate by Model and Prompt Condition\n(Claude Judge)", ylim=(0, 0.65))
fig.tight_layout()
fig.savefig(OUT_DIR / "F2b_over_deferral_claudejudge.png", bbox_inches="tight")
plt.close()
print("Saved F2b")

# ── F4: Premature closure by subset × condition (heatmap) ────────────────────
rowlevel["pc_binary"] = (rowlevel["premature_closure"] == "yes").astype(float)
SUBSET_LABELS = {
    "underspecified_low_risk":       "Underspecified\n(Low-Risk)",
    "underspecified_urgent":         "Underspecified\n(Urgent)",
    "hedging_reducible_uncertainty": "Hedging /\nUncertainty",
}
CONDITION_LABELS = {"baseline": "Baseline", "safety": "Safety", "clarification": "Clarification-First"}

heat_df = rowlevel[rowlevel["subset"] != "sufficient_context"].copy()
heat_df["subset_label"]    = heat_df["subset"].map(SUBSET_LABELS)
heat_df["condition_label"] = heat_df["condition"].map(CONDITION_LABELS)

pivot = heat_df.groupby(["subset_label", "condition_label"])["pc_binary"].mean().unstack()
pivot = pivot[["Baseline", "Safety", "Clarification-First"]]

fig, ax = plt.subplots(figsize=(7, 4))
im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=0.8, aspect="auto")
ax.set_xticks(range(3))
ax.set_xticklabels(["Baseline", "Safety", "Clarification-First"], fontsize=11)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=10)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        val = pivot.values[i, j]
        ax.text(j, i, f"{val*100:.1f}%", ha="center", va="center",
                color="white" if val > 0.5 else "black", fontsize=11, fontweight="bold")
plt.colorbar(im, ax=ax, label="Premature Closure Rate")
ax.set_title("Figure 4. Premature Closure Rate by Subset and Prompt Condition\n(GPT-5 Judge, excluding Sufficient Context)",
             fontsize=12, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "F4_subset_heatmap.png", bbox_inches="tight")
plt.close()
print("Saved F4")

# ── F5: GPT-5 vs Claude judge PC comparison (scatter) ────────────────────────
MODEL_LABEL_MAP = {"gpt": "GPT-5", "claude": "Claude", "gemini": "Gemini",
                   "llama": "Llama", "deepseek": "DeepSeek"}
COND_LABEL_MAP  = {"baseline": "Baseline", "safety": "Safety", "clarification": "Clarification-First"}

gpt_pc    = gpt_agg[["Model", "Condition", "Premature Closure Rate"]].copy()
claude_pc = claude_agg[["Model", "Condition", "Premature Closure"]].copy()
claude_pc.columns = ["Model", "Condition", "Premature Closure Rate"]

merged = gpt_pc.merge(claude_pc, on=["Model", "Condition"], suffixes=("_gpt", "_claude"))

fig, ax = plt.subplots(figsize=(6, 6))
for model in MODEL_ORDER:
    sub = merged[merged["Model"] == model]
    ax.scatter(sub["Premature Closure Rate_gpt"], sub["Premature Closure Rate_claude"],
               color=MODEL_COLORS[model], s=80, label=model, zorder=3)

lims = [0, 0.65]
ax.plot(lims, lims, "k--", alpha=0.4, linewidth=1, label="Perfect agreement")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_xlabel("GPT-5 Judge — Premature Closure Rate", fontsize=11)
ax.set_ylabel("Claude Judge — Premature Closure Rate", fontsize=11)
ax.set_title("Figure 5. Inter-Judge Comparison: Premature Closure Rate\n(Each point = one model × condition cell)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(OUT_DIR / "F5_judge_comparison.png", bbox_inches="tight")
plt.close()
print("Saved F5")

# ── F6: Behavior distribution by condition (stacked bar) ─────────────────────
BEHAVIOR_COLORS = {
    "ANSWERED":  "#4C72B0",
    "PARTIAL":   "#8172B3",
    "CLARIFIED": "#55A868",
    "ESCALATED": "#DD8452",
}
BEHAVIORS = ["ANSWERED", "PARTIAL", "CLARIFIED", "ESCALATED"]

rowlevel["condition_label"] = rowlevel["condition"].map(COND_LABEL_MAP)
cond_counts = rowlevel.groupby(["condition_label", "behavior"]).size().unstack(fill_value=0)
cond_pct = cond_counts.div(cond_counts.sum(axis=1), axis=0)
cond_pct = cond_pct.reindex(["Baseline", "Safety", "Clarification-First"])
cond_pct = cond_pct[[b for b in BEHAVIORS if b in cond_pct.columns]]

fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(3)
for behavior in BEHAVIORS:
    if behavior not in cond_pct.columns:
        continue
    vals = cond_pct[behavior].values
    bars = ax.bar(range(3), vals, bottom=bottom, label=behavior,
                  color=BEHAVIOR_COLORS[behavior], edgecolor="white", width=0.5)
    for j, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.04:
            ax.text(j, b + v/2, f"{v*100:.0f}%", ha="center", va="center",
                    color="white", fontsize=10, fontweight="bold")
    bottom += vals

ax.set_xticks(range(3))
ax.set_xticklabels(["Baseline", "Safety", "Clarification-First"], fontsize=11)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
ax.set_ylabel("Proportion of Responses")
ax.set_title("Figure 6. Response Behavior Distribution by Prompt Condition\n(GPT-5 Judge, all models combined)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(frameon=False, fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
fig.savefig(OUT_DIR / "F6_behavior_distribution.png", bbox_inches="tight")
plt.close()
print("Saved F6")

# ── T1: Aggregate results table — GPT-5 judge ────────────────────────────────
t1 = gpt_agg[["Model", "Condition", "N",
              "Premature Closure Rate", "Over-Deferral Rate",
              "Mean Rubric Score"]].copy()
for col in ["Premature Closure Rate", "Over-Deferral Rate", "Mean Rubric Score"]:
    t1[col] = t1[col].apply(lambda x: f"{x*100:.1f}%")
t1.to_csv(OUT_DIR / "T1_results_gpt5judge.csv", index=False)
print("Saved T1")

# ── T2: Aggregate results table — Claude judge ────────────────────────────────
t2 = claude_agg[["Model", "Condition", "N",
                  "Premature Closure", "Over-Deferral",
                  "Mean Rubric Score"]].copy()
t2.columns = ["Model", "Condition", "N",
              "Premature Closure Rate", "Over-Deferral Rate", "Mean Rubric Score"]
for col in ["Premature Closure Rate", "Over-Deferral Rate", "Mean Rubric Score"]:
    t2[col] = t2[col].apply(lambda x: f"{x*100:.1f}%")
t2.to_csv(OUT_DIR / "T2_results_claudejudge.csv", index=False)
print("Saved T2")

# ── T3: Inter-rater agreement ─────────────────────────────────────────────────
from sklearn.metrics import cohen_kappa_score
kappa_b  = cohen_kappa_score(agreement["gpt_behavior"],  agreement["claude_behavior"])
kappa_pc = cohen_kappa_score(agreement["gpt_pc"].fillna("na"), agreement["claude_pc"].fillna("na"))
agree_b  = agreement["behavior_agrees"].mean()
agree_pc = agreement["pc_agrees"].mean()

agree_by_model = agreement.groupby("model")["behavior_agrees"].mean().reset_index()
agree_by_model.columns = ["Model", "Behavior Agreement"]
agree_by_model["Model"] = agree_by_model["Model"].map(MODEL_LABEL_MAP)
agree_by_model["Behavior Agreement"] = agree_by_model["Behavior Agreement"].apply(lambda x: f"{x*100:.1f}%")

t3_summary = pd.DataFrame([
    {"Metric": "Overall Behavior Agreement",          "Value": f"{agree_b*100:.1f}%"},
    {"Metric": "Overall Premature Closure Agreement", "Value": f"{agree_pc*100:.1f}%"},
    {"Metric": "Cohen's Kappa (Behavior)",            "Value": f"{kappa_b:.3f}"},
    {"Metric": "Cohen's Kappa (Premature Closure)",   "Value": f"{kappa_pc:.3f}"},
    {"Metric": "Total Paired Evaluations",            "Value": str(len(agreement))},
])
t3_summary.to_csv(OUT_DIR / "T3_judge_agreement.csv", index=False)
agree_by_model.to_csv(OUT_DIR / "T3b_agreement_by_model.csv", index=False)
print("Saved T3")

print(f"\nAll outputs saved to {OUT_DIR}/")
print("Figures: F1–F6  |  Tables: T1–T3")
