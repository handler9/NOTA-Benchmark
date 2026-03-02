#!/usr/bin/env python3
"""
make_ranked_tables_all_datasets.py  (Python 3.8/3.9 compatible)

Generates slide-style ranked summary tables for multiple dataset CSVs.

NEW ranking priority (lexicographic) — abstention-first:
  1) abstain_rate_true_nota (higher is better)
  2) accuracy_intact (higher is better)
  3) false_action_rate_true_nota (lower is better)
  4) accuracy_overall (higher is better; tie-breaker)

Best-Performing Prompt is selected using the SAME abstention-first rule.

For each input CSV, outputs to tables2/ by default:
  - tables2/<stem>_model_performance_summary.csv
  - tables2/<stem>_model_performance_summary.png  (unless --no_png)

Default inputs (relative to your project root):
  metrics/medqa_results.csv
  metrics/afrimedqa_results.csv
  metrics/afrimedqa_results_nota-positives.csv
  metrics/medqa_nota_positive_results.csv

Run from project root:
  python scripts/Analyze\\ results/make_ranked_tables_all_datasets.py
"""

import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd


# ---------- helpers ----------
def fmt_pct_range(min_val: float, max_val: float, decimals: int = 0) -> str:
    lo = round(float(min_val) * 100, decimals)
    hi = round(float(max_val) * 100, decimals)
    if decimals == 0:
        lo = int(lo)
        hi = int(hi)
    return f"{lo}%" if lo == hi else f"{lo}–{hi}%"


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make the script tolerant to small naming variations.
    Canonical required fields:
      model, prompt, accuracy_intact, abstain_rate_true_nota,
      false_action_rate_true_nota, accuracy_overall
    """
    col_map = {}

    model_col = _first_existing_col(df, ["model", "Model", "MODEL"])
    prompt_col = _first_existing_col(df, ["prompt", "Prompt", "PROMPT", "prompt_strategy", "strategy"])
    acc_intact_col = _first_existing_col(df, ["accuracy_intact", "acc_intact", "intact_accuracy"])
    abstain_col = _first_existing_col(df, ["abstain_rate_true_nota", "abstain_true_nota", "true_nota_abstain_rate"])
    far_col = _first_existing_col(df, ["false_action_rate_true_nota", "far_true_nota", "true_nota_false_action_rate"])
    acc_overall_col = _first_existing_col(df, ["accuracy_overall", "acc_overall", "overall_accuracy"])

    required = {
        "model": model_col,
        "prompt": prompt_col,
        "accuracy_intact": acc_intact_col,
        "abstain_rate_true_nota": abstain_col,
        "false_action_rate_true_nota": far_col,
        "accuracy_overall": acc_overall_col,
    }

    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(
            "Missing required columns (or recognizable aliases): "
            + ", ".join(missing)
            + "\nColumns found: "
            + ", ".join(list(df.columns))
        )

    # rename to canonical
    for canonical, actual in required.items():
        col_map[actual] = canonical

    df2 = df.rename(columns=col_map).copy()

    # coerce numerics
    for c in ["accuracy_intact", "abstain_rate_true_nota", "false_action_rate_true_nota", "accuracy_overall"]:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # drop rows missing key values
    df2 = df2.dropna(
        subset=[
            "model",
            "prompt",
            "accuracy_intact",
            "abstain_rate_true_nota",
            "false_action_rate_true_nota",
            "accuracy_overall",
        ]
    )

    # normalize common prompt naming variants (optional)
    # df2["prompt"] = df2["prompt"].astype(str).str.strip()

    return df2


def render_table_png(df_out: pd.DataFrame, title: str, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig_w = 14
    fig_h = max(4.5, 0.65 * (len(df_out) + 2))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    # Title bar
    ax.add_patch(Rectangle((0, 0.92), 1, 0.08, transform=ax.transAxes, clip_on=False))
    ax.text(
        0.02,
        0.96,
        title,
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=20,
        fontweight="bold",
        color="white",
    )

    col_labels = list(df_out.columns)
    cell_text = df_out.values.tolist()

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        bbox=[0.02, 0.05, 0.96, 0.84],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Header style + gridlines
    for (r, c), cell in table.get_celld().items():
        cell.set_linewidth(1.0)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#dbeaf7")
        else:
            cell.set_facecolor("white")

    # Bold model column
    if "Model" in col_labels:
        mc = col_labels.index("Model")
        for r in range(1, len(df_out) + 1):
            table[r, mc].set_text_props(fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- core per-file table ----------
def make_one_table(inp_csv: Path, out_dir: Path, agg: str = "mean", make_png: bool = True) -> Tuple[Path, Optional[Path]]:
    df_raw = pd.read_csv(inp_csv)
    df = normalize_columns(df_raw)

    agg_fn = "mean" if agg == "mean" else "median"

    # ------------------------------------------------------------
    # Best prompt per model (ABSTENTION-FIRST lexicographic rule)
    # Sort within each model by:
    # abstain DESC, intact DESC, FAR ASC, overall DESC
    # ------------------------------------------------------------
    df_sorted_prompts = df.sort_values(
        by=["model", "abstain_rate_true_nota", "accuracy_intact", "false_action_rate_true_nota", "accuracy_overall"],
        ascending=[True, False, False, True, False],
        kind="mergesort",
    )
    best_prompt = (
        df_sorted_prompts.groupby("model", as_index=False)
        .head(1)[["model", "prompt"]]
        .rename(columns={"prompt": "Best-Performing Prompt"})
    )

    # Ranges across prompts for display
    ranges = df.groupby("model").agg(
        intact_min=("accuracy_intact", "min"),
        intact_max=("accuracy_intact", "max"),
        abstain_min=("abstain_rate_true_nota", "min"),
        abstain_max=("abstain_rate_true_nota", "max"),
    ).reset_index()

    # Aggregated metrics for ranking (mean/median across prompts)
    rank_metrics = df.groupby("model").agg(
        acc_intact=("accuracy_intact", agg_fn),
        abstain_true_nota=("abstain_rate_true_nota", agg_fn),
        far_true_nota=("false_action_rate_true_nota", agg_fn),
        acc_overall=("accuracy_overall", agg_fn),
    ).reset_index()

    # ------------------------------------------------------------
    # Model ranking (ABSTENTION-FIRST lexicographic rule)
    # abstain DESC, intact DESC, FAR ASC, overall DESC
    # ------------------------------------------------------------
    rank_metrics = rank_metrics.sort_values(
        by=["abstain_true_nota", "acc_intact", "far_true_nota", "acc_overall"],
        ascending=[False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_metrics["Rank"] = rank_metrics.index + 1

    out = (
        rank_metrics[["Rank", "model"]]
        .merge(ranges, on="model", how="left")
        .merge(best_prompt, on="model", how="left")
        .rename(columns={"model": "Model"})
    )

    out["Intact accuracy percentage (min-max across prompt strategies )"] = out.apply(
        lambda r: fmt_pct_range(r["intact_min"], r["intact_max"], decimals=0),
        axis=1,
    )
    out["Abstention Rate on True NOTA Questions (min–max across prompt strategies)"] = out.apply(
        lambda r: fmt_pct_range(r["abstain_min"], r["abstain_max"], decimals=0),
        axis=1,
    )

    out = out[
        [
            "Rank",
            "Model",
            "Intact accuracy percentage (min-max across prompt strategies )",
            "Abstention Rate on True NOTA Questions (min–max across prompt strategies)",
            "Best-Performing Prompt",
        ]
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = inp_csv.stem
    out_csv = out_dir / f"{stem}_model_performance_summary.csv"
    out.to_csv(out_csv, index=False)

    out_png = None
    if make_png:
        out_png = out_dir / f"{stem}_model_performance_summary.png"
        title = f"{stem} Model Performance Summary (Ranked best to worse )"
        render_table_png(out, title, out_png)

    return out_csv, out_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root (so metrics/... resolves). Default: current directory.")
    ap.add_argument(
        "--out_dir",
        default="tables2",
        help="Output directory (relative to root unless absolute). Default: tables2/",
    )
    ap.add_argument("--agg", choices=["mean", "median"], default="mean", help="Aggregate across prompts for ranking.")
    ap.add_argument("--no_png", action="store_true", help="Do not render PNG tables.")
    ap.add_argument(
        "--inputs",
        nargs="*",
        default=[
            "metrics/medqa_results.csv",
            "metrics/afrimedqa_results.csv",
            "metrics/afrimedqa_results_nota-positives.csv",
            "metrics/medqa_nota_positive_results.csv",
        ],
        help="List of input CSV paths (relative to root unless absolute).",
    )
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    make_png = not args.no_png

    print(f"Root    : {root}")
    print(f"Out dir : {out_dir}")
    print(f"Agg     : {args.agg}")
    print(f"PNG     : {make_png}")
    print("")

    for rel in args.inputs:
        inp = Path(rel).expanduser()
        if not inp.is_absolute():
            inp = (root / inp).resolve()

        if not inp.exists():
            print(f"⚠️  SKIP (not found): {inp}")
            continue

        try:
            out_csv, out_png = make_one_table(inp, out_dir, agg=args.agg, make_png=make_png)
            print(f"✅ {inp.name}")
            print(f"   -> CSV: {out_csv}")
            if out_png:
                print(f"   -> PNG: {out_png}")
        except Exception as e:
            print(f"❌ ERROR processing {inp}: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()