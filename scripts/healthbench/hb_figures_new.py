"""
HealthBench Figures — New Pipeline
Produces publication-ready figures saved to metrics/figures/.

Figures:
  1. hb_pc_by_model.png          — Premature closure rate: baseline vs safety, all models (HB)
  2. hb_pc_by_subset.png         — Premature closure rate by subset × model (HB, baseline only)
  3. hb_rubric_by_model.png      — Mean rubric score %: baseline vs safety (HB)
  4. hb_behavior_dist.png        — Behavior distribution stacked bars (HB, baseline)
  5. redteam_pc_by_model.png     — Red-team premature closure: baseline vs safety
  6. redteam_rubric_by_model.png — Red-team rubric score: baseline vs safety
  7. medqa_accuracy.png          — MedQA overall accuracy by model × prompt
  8. afrimedqa_accuracy.png      — AfriMedQA overall accuracy by model × prompt

Run from project root:
  python -u scripts/healthbench/hb_figures_new.py
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path("metrics/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

MODEL_ORDER  = ["GPT-5.4", "Claude Opus 4.7", "Grok 3", "Gemini 2.5 Pro", "DeepSeek R1"]
MODEL_COLORS = {
    "GPT-5.4":        "#4C72B0",
    "Claude Opus 4.7":"#DD8452",
    "Grok 3":         "#55A868",
    "Gemini 2.5 Pro": "#C44E52",
    "DeepSeek R1":    "#8172B2",
}
COND_COLORS  = {"baseline": "#4C72B0", "safety": "#DD8452"}
SUBSET_LABELS = {
    "underspecified_low_risk":       "Underspecified\n(low risk)",
    "underspecified_urgent":         "Underspecified\n(urgent)",
    "sufficient_context":            "Sufficient\ncontext",
    "hedging_reducible_uncertainty": "Hedging /\nreducible uncertainty",
}
PROMPT_LABELS = {
    "baseline":        "Baseline",
    "safety":          "Safety prompt",
    "think":           "Think-then-decide",
    "doublecheck":     "Answer-then-double-check",
    "answer-then-double-check": "Answer-then-double-check",
    "safety-prompt":   "Safety prompt",
    "think-then-decide": "Think-then-decide",
}
PROMPT_ORDER = ["Baseline", "Safety prompt", "Think-then-decide", "Answer-then-double-check"]
PROMPT_COLORS = {
    "Baseline":                    "#4C72B0",
    "Safety prompt":               "#DD8452",
    "Think-then-decide":           "#55A868",
    "Answer-then-double-check":    "#C44E52",
}


def savefig(name):
    p = FIG_DIR / name
    plt.savefig(p, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {p}")


# ══════════════════════════════════════════════════════════════════════
# Figure 1 — HB: Premature closure baseline vs safety (grouped bars)
# ══════════════════════════════════════════════════════════════════════
def fig_hb_pc_by_model():
    agg = pd.read_csv("metrics/hb_aggregate_overall.csv")
    agg = agg[agg["model"].isin(MODEL_ORDER)].copy()

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cond in enumerate(["baseline", "safety"]):
        vals = [agg.loc[(agg["model"] == m) & (agg["condition"] == cond),
                        "premature_closure_rate"].values[0]
                if len(agg.loc[(agg["model"] == m) & (agg["condition"] == cond)]) > 0
                else 0
                for m in MODEL_ORDER]
        bars = ax.bar(x + ci * w - w / 2, vals, w, label=cond.capitalize(),
                      color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Premature Closure Rate (%)")
    ax.set_title("HealthBench: Premature Closure Rate by Model and Prompt Condition")
    ax.legend(title="Condition")
    ax.set_ylim(0, 60)
    savefig("hb_pc_by_model.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 2 — HB: PC rate by subset × model (baseline only, heatmap)
# ══════════════════════════════════════════════════════════════════════
def fig_hb_pc_by_subset():
    agg = pd.read_csv("metrics/hb_aggregate.csv")
    base = agg[agg["condition"] == "baseline"].copy()

    subsets = list(SUBSET_LABELS.keys())
    matrix = pd.DataFrame(index=MODEL_ORDER, columns=subsets, dtype=float)
    for m in MODEL_ORDER:
        for s in subsets:
            row = base[(base["model"] == m) & (base["subset"] == s)]
            matrix.loc[m, s] = row["premature_closure_rate"].values[0] if len(row) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(10, 4.5))
    im = ax.imshow(matrix.values.astype(float), cmap="RdYlGn_r", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels([SUBSET_LABELS[s] for s in subsets], fontsize=10)
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER, fontsize=10)
    ax.set_title("HealthBench: Premature Closure Rate by Model and Question Subset (Baseline)")

    for i, m in enumerate(MODEL_ORDER):
        for j, s in enumerate(subsets):
            v = matrix.loc[m, s]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=10, color="black" if v < 70 else "white",
                        fontweight="bold")

    plt.colorbar(im, ax=ax, label="Premature Closure %", shrink=0.8)
    savefig("hb_pc_by_subset.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 3 — HB: Rubric score baseline vs safety
# ══════════════════════════════════════════════════════════════════════
def fig_hb_rubric_by_model():
    agg = pd.read_csv("metrics/hb_aggregate_overall.csv")
    agg = agg[agg["model"].isin(MODEL_ORDER)].copy()

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cond in enumerate(["baseline", "safety"]):
        vals = [agg.loc[(agg["model"] == m) & (agg["condition"] == cond),
                        "mean_rubric_score_pct"].values[0]
                if len(agg.loc[(agg["model"] == m) & (agg["condition"] == cond)]) > 0
                else 0
                for m in MODEL_ORDER]
        bars = ax.bar(x + ci * w - w / 2, vals, w, label=cond.capitalize(),
                      color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Mean Rubric Score (%)")
    ax.set_title("HealthBench: Mean Rubric Score by Model and Prompt Condition")
    ax.legend(title="Condition")
    ax.set_ylim(0, 80)
    savefig("hb_rubric_by_model.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 4 — HB: Behavior distribution (stacked bars, baseline)
# ══════════════════════════════════════════════════════════════════════
def fig_hb_behavior_dist():
    agg = pd.read_csv("metrics/hb_aggregate_overall.csv")
    base = agg[agg["condition"] == "baseline"].copy()
    base = base[base["model"].isin(MODEL_ORDER)].set_index("model").loc[MODEL_ORDER]

    behs   = ["pct_answered", "pct_partial", "pct_escalated", "pct_clarified"]
    labels = ["Answered", "Partial", "Escalated", "Clarified"]
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x   = np.arange(len(MODEL_ORDER))
    bot = np.zeros(len(MODEL_ORDER))

    for col, lab, col_c in zip(behs, labels, colors):
        vals = base[col].values.astype(float)
        ax.bar(x, vals, bottom=bot, label=lab, color=col_c, alpha=0.85, edgecolor="white")
        bot += vals

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("% of Responses")
    ax.set_title("HealthBench: Response Behavior Distribution (Baseline)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1))
    ax.set_ylim(0, 105)
    savefig("hb_behavior_dist.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 5 — Red-team: PC rate baseline vs safety
# ══════════════════════════════════════════════════════════════════════
def fig_redteam_pc_by_model():
    agg = pd.read_csv("metrics/hb_redteam_aggregate.csv")
    agg = agg[agg["model"].isin(MODEL_ORDER)].copy()

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cond in enumerate(["baseline", "safety"]):
        vals = [agg.loc[(agg["model"] == m) & (agg["condition"] == cond),
                        "premature_closure_rate"].values[0]
                if len(agg.loc[(agg["model"] == m) & (agg["condition"] == cond)]) > 0
                else 0
                for m in MODEL_ORDER]
        bars = ax.bar(x + ci * w - w / 2, vals, w, label=cond.capitalize(),
                      color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Premature Closure Rate (%)")
    ax.set_title("Red-Team Queries: Premature Closure Rate by Model and Prompt Condition")
    ax.legend(title="Condition")
    ax.set_ylim(0, 110)
    savefig("redteam_pc_by_model.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 6 — Red-team: Rubric score baseline vs safety
# ══════════════════════════════════════════════════════════════════════
def fig_redteam_rubric_by_model():
    agg = pd.read_csv("metrics/hb_redteam_aggregate.csv")
    agg = agg[agg["model"].isin(MODEL_ORDER)].copy()

    x = np.arange(len(MODEL_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))

    for ci, cond in enumerate(["baseline", "safety"]):
        vals = [agg.loc[(agg["model"] == m) & (agg["condition"] == cond),
                        "mean_rubric_score_pct"].values[0]
                if len(agg.loc[(agg["model"] == m) & (agg["condition"] == cond)]) > 0
                else 0
                for m in MODEL_ORDER]
        bars = ax.bar(x + ci * w - w / 2, vals, w, label=cond.capitalize(),
                      color=COND_COLORS[cond], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.5 if v >= 0 else -3),
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Mean Rubric Score (%)")
    ax.set_title("Red-Team Queries: Mean Rubric Score by Model and Prompt Condition")
    ax.legend(title="Condition")
    savefig("redteam_rubric_by_model.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 7 — MedQA: Overall accuracy by model × prompt
# ══════════════════════════════════════════════════════════════════════
def fig_medqa_accuracy():
    path = Path("metrics/medqa_results_new.csv")
    if not path.exists():
        print("  ⚠️  metrics/medqa_results_new.csv not found, skipping")
        return

    df = pd.read_csv(path)
    df["prompt_pretty"] = df["prompt"].map(PROMPT_LABELS).fillna(df["prompt"])
    df = df[df["model"].isin(MODEL_ORDER)]

    prompts = [p for p in PROMPT_ORDER if p in df["prompt_pretty"].unique()]
    x = np.arange(len(MODEL_ORDER))
    w = 0.8 / len(prompts)

    fig, ax = plt.subplots(figsize=(11, 5))
    for pi, prompt in enumerate(prompts):
        sub = df[df["prompt_pretty"] == prompt].set_index("model")
        vals = []
        for m in MODEL_ORDER:
            if m in sub.index:
                vals.append(sub.loc[m, "accuracy_overall"] * 100
                            if sub.loc[m, "accuracy_overall"] <= 1
                            else sub.loc[m, "accuracy_overall"])
            else:
                vals.append(0)
        offset = pi * w - (len(prompts) - 1) * w / 2
        bars = ax.bar(x + offset, vals, w, label=prompt,
                      color=PROMPT_COLORS.get(prompt, f"C{pi}"), alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("MedQA-NOTA: Overall Accuracy by Model and Prompt Condition")
    ax.legend(title="Prompt", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    savefig("medqa_accuracy.png")


# ══════════════════════════════════════════════════════════════════════
# Figure 8 — AfriMedQA: Overall accuracy by model × prompt
# ══════════════════════════════════════════════════════════════════════
def fig_afrimedqa_accuracy():
    path = Path("metrics/afrimedqa_results_new.csv")
    if not path.exists():
        print("  ⚠️  metrics/afrimedqa_results_new.csv not found, skipping")
        return

    df = pd.read_csv(path)
    df["prompt_pretty"] = df["prompt"].map(PROMPT_LABELS).fillna(df["prompt"])
    df = df[df["model"].isin(MODEL_ORDER)]

    prompts = [p for p in PROMPT_ORDER if p in df["prompt_pretty"].unique()]
    x = np.arange(len(MODEL_ORDER))
    w = 0.8 / len(prompts)

    fig, ax = plt.subplots(figsize=(11, 5))
    for pi, prompt in enumerate(prompts):
        sub = df[df["prompt_pretty"] == prompt].set_index("model")
        vals = []
        for m in MODEL_ORDER:
            if m in sub.index:
                v = sub.loc[m, "accuracy_overall"]
                vals.append(v * 100 if v <= 1 else v)
            else:
                vals.append(0)
        offset = pi * w - (len(prompts) - 1) * w / 2
        bars = ax.bar(x + offset, vals, w, label=prompt,
                      color=PROMPT_COLORS.get(prompt, f"C{pi}"), alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{v:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Overall Accuracy (%)")
    ax.set_title("AfriMedQA-NOTA: Overall Accuracy by Model and Prompt Condition")
    ax.legend(title="Prompt", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set_ylim(0, 100)
    savefig("afrimedqa_accuracy.png")


# ══════════════════════════════════════════════════════════════════════
# Run all figures
# ══════════════════════════════════════════════════════════════════════
print("Generating figures...")
fig_hb_pc_by_model()
fig_hb_pc_by_subset()
fig_hb_rubric_by_model()
fig_hb_behavior_dist()
fig_redteam_pc_by_model()
fig_redteam_rubric_by_model()
fig_medqa_accuracy()
fig_afrimedqa_accuracy()
print(f"\n✅ All figures saved to {FIG_DIR}/")
