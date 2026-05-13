"""
Generate revised figures per collaborator comments.

Outputs:
  metrics/figures/F_structured_tradeoff.png
      Scatter: false-action rate vs intact accuracy, with baseline and
      safety-prompt points connected for each model.
  metrics/figures/F_scatter_urgent_vs_lowrisk.png
      X-Y scatter: underspecified urgent PC rate vs underspecified low-risk PC rate,
      baseline-to-safety arrows.
  metrics/figures/F_scatter_pc_vs_rubric.png
      X-Y scatter: overall HealthBench PC rate vs rubric score,
      baseline-to-safety arrows.
  metrics/figures/F_summary_table.png
      Consolidated model-by-dataset summary table for baseline and safety.

Run from repo root:
  python scripts/healthbench/make_figures_revised.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


OUT_DIR = Path("metrics/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = [
    "GPT-5.4",
    "Claude Opus 4.7",
    "Grok 3",
    "Gemini 2.5 Pro",
    "DeepSeek R1",
]

MODEL_COLORS = {
    "GPT-5.4": "#2F6B9A",
    "Claude Opus 4.7": "#5B8E47",
    "Grok 3": "#D9852B",
    "Gemini 2.5 Pro": "#8E5EA2",
    "DeepSeek R1": "#C94C4C",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "figure.dpi": 150,
    }
)


def savefig(name: str) -> None:
    path = OUT_DIR / name
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def load_metrics():
    hb_subset = pd.read_csv("metrics/hb_aggregate.csv")
    hb_overall = pd.read_csv("metrics/hb_aggregate_overall.csv")
    redteam = pd.read_csv("metrics/hb_redteam_aggregate.csv")
    medqa = pd.read_csv("metrics/medqa_results_new.csv")
    afrimedqa = pd.read_csv("metrics/afrimedqa_results_new.csv")
    return hb_subset, hb_overall, redteam, medqa, afrimedqa


def value(df, model, condition, column, subset=None, prompt=None):
    rows = df[df["model"].eq(model)].copy()
    if condition is not None and "condition" in rows.columns:
        rows = rows[rows["condition"].eq(condition)]
    if subset is not None:
        rows = rows[rows["subset"].eq(subset)]
    if prompt is not None:
        rows = rows[rows["prompt"].eq(prompt)]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][column])


def draw_model_legend(ax, loc="upper left", bbox_to_anchor=(1.02, 1.0)):
    handles = [
        mpatches.Patch(color=MODEL_COLORS[model], label=model)
        for model in MODEL_ORDER
    ]
    baseline = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="#555555",
        linestyle="None",
        markersize=7,
        label="Baseline",
    )
    safety = plt.Line2D(
        [0],
        [0],
        marker="o",
        color="#555555",
        markerfacecolor="white",
        markeredgewidth=1.8,
        linestyle="None",
        markersize=7,
        label="Safety prompt",
    )
    ax.legend(
        handles=handles + [baseline, safety],
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )


def prompt_value(df, model, prompt, column):
    rows = df[(df["model"].eq(model)) & (df["prompt"].eq(prompt))]
    if rows.empty:
        return np.nan
    return float(rows.iloc[0][column])


def fig_structured_tradeoff(medqa, afrimedqa):
    """Baseline-to-safety movement in structured NOTA tasks."""
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 4.9), sharey=True)
    datasets = [("MedQA", medqa), ("AfriMed-QA", afrimedqa)]

    for ax, (name, df) in zip(axes, datasets):
        for model in MODEL_ORDER:
            color = MODEL_COLORS[model]
            false_action_base = (
                prompt_value(df, model, "baseline", "false_action_rate_true_nota") * 100
            )
            false_action_safety = (
                prompt_value(df, model, "safety-prompt", "false_action_rate_true_nota") * 100
            )
            intact_base = prompt_value(df, model, "baseline", "accuracy_intact") * 100
            intact_safety = (
                prompt_value(df, model, "safety-prompt", "accuracy_intact") * 100
            )

            ax.plot(
                [false_action_base, false_action_safety],
                [intact_base, intact_safety],
                color=color,
                lw=1.8,
                alpha=0.85,
                zorder=3,
            )
            ax.scatter(
                false_action_base,
                intact_base,
                s=95,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
            ax.scatter(
                false_action_safety,
                intact_safety,
                s=95,
                facecolors="white",
                edgecolors=color,
                linewidths=1.9,
                zorder=5,
            )

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("False Action Rate on NOTA Items (%)")
        ax.set_xlim(20, 85)
        ax.set_ylim(70, 95)
        ax.text(
            22,
            93.2,
            "Lower false action,\nhigher intact accuracy",
            fontsize=8.5,
            color="#555555",
        )

    axes[0].set_ylabel("Intact Accuracy (%)")
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="-",
            markersize=8,
            color=MODEL_COLORS[model],
            markerfacecolor=MODEL_COLORS[model],
            markeredgecolor="white",
            label=model,
        )
        for model in MODEL_ORDER
    ]
    condition_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            linestyle="None",
            markersize=8,
            markerfacecolor="#555555",
            markeredgecolor="white",
            label="Baseline",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="#555555",
            linestyle="None",
            markersize=8,
            markerfacecolor="white",
            markeredgecolor="#555555",
            markeredgewidth=1.8,
            label="Safety prompt",
        ),
    ]
    axes[1].legend(
        handles=model_handles + condition_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )
    fig.suptitle(
        "Effect of Safety Prompting on False Action and Intact Accuracy",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Each line connects baseline and safety-prompt results for one model. "
        "False action is selecting an answer on unanswerable NOTA items; "
        "open circles denote safety-prompt results.",
        ha="center",
        fontsize=8.5,
        color="#555555",
    )
    savefig("F_structured_tradeoff.png")


def fig_lowrisk_vs_urgent(hb_subset):
    """Reviewer-requested scatter replacing the larger subset figure."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        x0 = value(
            hb_subset,
            model,
            "baseline",
            "premature_closure_rate",
            subset="underspecified_urgent",
        )
        y0 = value(
            hb_subset,
            model,
            "baseline",
            "premature_closure_rate",
            subset="underspecified_low_risk",
        )
        x1 = value(
            hb_subset,
            model,
            "safety",
            "premature_closure_rate",
            subset="underspecified_urgent",
        )
        y1 = value(
            hb_subset,
            model,
            "safety",
            "premature_closure_rate",
            subset="underspecified_low_risk",
        )

        ax.scatter(x0, y0, s=80, color=color, zorder=4)
        ax.scatter(
            x1,
            y1,
            s=80,
            facecolors="white",
            edgecolors=color,
            linewidths=2,
            zorder=5,
        )
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.8, zorder=3)
    ax.set_xlabel("Premature Closure Rate: Underspecified Urgent (%)")
    ax.set_ylabel("Premature Closure Rate: Underspecified Low Risk (%)")
    ax.set_title(
        "Missing Context vs. Urgency Sensitivity\nBaseline to Safety Prompt, HealthBench",
        pad=10,
    )
    ax.set_xlim(0, 42)
    ax.set_ylim(55, 100)
    ax.text(
        1,
        96.5,
        "Lower-left is better",
        fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD"),
    )
    draw_model_legend(ax, loc="lower right", bbox_to_anchor=(0.98, 0.02))
    savefig("F_scatter_urgent_vs_lowrisk.png")


def fig_pc_vs_rubric(hb_overall):
    """Reviewer-requested scatter of safety movement in the main open-ended task."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for model in MODEL_ORDER:
        color = MODEL_COLORS[model]
        x0 = value(hb_overall, model, "baseline", "premature_closure_rate")
        y0 = value(hb_overall, model, "baseline", "mean_rubric_score_pct")
        x1 = value(hb_overall, model, "safety", "premature_closure_rate")
        y1 = value(hb_overall, model, "safety", "mean_rubric_score_pct")

        ax.scatter(x0, y0, s=80, color=color, zorder=4)
        ax.scatter(
            x1,
            y1,
            s=80,
            facecolors="white",
            edgecolors=color,
            linewidths=2,
            zorder=5,
        )
        ax.plot([x0, x1], [y0, y1], color=color, lw=1.8, zorder=3)
    ax.set_xlabel("Premature Closure Rate (%)")
    ax.set_ylabel("Mean Rubric Score (%)")
    ax.set_title(
        "Safety Prompting Reduces Premature Closure\nWhile Preserving Rubric Performance",
        pad=10,
    )
    ax.set_xlim(12, 48)
    ax.set_ylim(40, 66)
    ax.text(
        13,
        64,
        "Upper-left is better",
        fontsize=9,
        color="#555555",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#DDDDDD"),
    )
    draw_model_legend(ax, loc="lower left", bbox_to_anchor=(0.02, 0.02))
    savefig("F_scatter_pc_vs_rubric.png")


def table_metric_rows(hb_overall, redteam, medqa, afrimedqa, structured_metric="abstention"):
    rows = []
    for model in MODEL_ORDER:
        medqa_base = value(medqa, model, None, "abstain_rate_true_nota", prompt="baseline") * 100
        medqa_safety = (
            value(medqa, model, None, "abstain_rate_true_nota", prompt="safety-prompt") * 100
        )
        afri_base = (
            value(afrimedqa, model, None, "abstain_rate_true_nota", prompt="baseline") * 100
        )
        afri_safety = (
            value(afrimedqa, model, None, "abstain_rate_true_nota", prompt="safety-prompt") * 100
        )
        if structured_metric == "pc":
            medqa_base = 100 - medqa_base
            medqa_safety = 100 - medqa_safety
            afri_base = 100 - afri_base
            afri_safety = 100 - afri_safety
        hb_base = value(hb_overall, model, "baseline", "premature_closure_rate")
        hb_safety = value(hb_overall, model, "safety", "premature_closure_rate")
        rt_base = value(redteam, model, "baseline", "premature_closure_rate")
        rt_safety = value(redteam, model, "safety", "premature_closure_rate")

        rows.append(
            [
                model,
                f"{medqa_base:.0f}",
                f"{medqa_safety:.0f}",
                f"{afri_base:.0f}",
                f"{afri_safety:.0f}",
                f"{hb_base:.0f}",
                f"{hb_safety:.0f}",
                f"{rt_base:.0f}",
                f"{rt_safety:.0f}",
            ]
        )
    return rows


def fig_summary_table(hb_overall, redteam, medqa, afrimedqa, structured_metric="abstention"):
    """Compact table to replace the overstuffed multi-panel summary figure."""
    rows = table_metric_rows(hb_overall, redteam, medqa, afrimedqa, structured_metric)
    headers = [
        "Model",
        "Baseline",
        "Safety",
        "Baseline",
        "Safety",
        "Baseline",
        "Safety",
        "Baseline",
        "Safety",
    ]

    fig, ax = plt.subplots(figsize=(11.4, 3.9))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colWidths=[0.20, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.75)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D0D0D0")
        cell.set_linewidth(0.7)
        if r == 0:
            cell.set_facecolor("#37474F")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == 0:
            cell.set_text_props(ha="left", fontweight="bold")
            cell.set_facecolor("#F7F7F7")
        elif structured_metric == "abstention" and c in (2, 4):
            cell.set_facecolor("#E8F5E9")
        elif c in (2, 4, 6, 8):
            cell.set_facecolor("#FFF8E1")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFAFA")

    if structured_metric == "pc":
        group_labels = [
            ("MedQA\nPC Rate (%)", 0.295, "Lower is better"),
            ("AfriMed-QA\nPC Rate (%)", 0.475, "Lower is better"),
            ("HealthBench\nPC Rate (%)", 0.655, "Lower is better"),
            ("Adversarial\nPC Rate (%)", 0.835, "Lower is better"),
        ]
        caption = (
            "All values are premature closure (PC) rates in percentages; lower values indicate better performance. "
            "For MedQA and AfriMed-QA, PC is calculated as 100 - abstention rate on unanswerable items."
        )
        out_name = "F_summary_table_pc.png"
    else:
        group_labels = [
            ("MedQA\nAbstention (%)", 0.295, "Higher is better"),
            ("AfriMed-QA\nAbstention (%)", 0.475, "Higher is better"),
            ("HealthBench\nPC Rate (%)", 0.655, "Lower is better"),
            ("Adversarial\nPC Rate (%)", 0.835, "Lower is better"),
        ]
        caption = (
            "All values are percentages. MedQA and AfriMed-QA show abstention on unanswerable items; "
            "HealthBench and adversarial columns show premature closure (PC) rates."
        )
        out_name = "F_summary_table.png"

    for label, x, sublabel in group_labels:
        ax.text(x, 0.91, label, transform=ax.transAxes, ha="center", va="bottom", fontweight="bold")
        ax.text(x, 0.865, sublabel, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="#555555")

    ax.set_title(
        "Premature Closure Across Structured and Open-Ended Evaluations",
        pad=30,
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.025,
        caption,
        ha="center",
        fontsize=8,
        color="#555555",
    )
    savefig(out_name)


def main():
    hb_subset, hb_overall, redteam, medqa, afrimedqa = load_metrics()
    fig_structured_tradeoff(medqa, afrimedqa)
    fig_lowrisk_vs_urgent(hb_subset)
    fig_pc_vs_rubric(hb_overall)
    fig_summary_table(hb_overall, redteam, medqa, afrimedqa, structured_metric="abstention")
    fig_summary_table(hb_overall, redteam, medqa, afrimedqa, structured_metric="pc")


if __name__ == "__main__":
    main()
