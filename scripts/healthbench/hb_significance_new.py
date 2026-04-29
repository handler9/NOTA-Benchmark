"""
Statistical Significance Testing for HealthBench Results
Tests:
  1. Baseline vs. safety prompt (per model) — paired McNemar's test on premature closure
  2. Between-model comparisons at baseline — pairwise McNemar's tests
  3. Rubric score differences — paired Wilcoxon signed-rank test

Covers both HB (861q) and red-team (191q).

Run from project root:
  python -u scripts/healthbench/hb_significance_new.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar

OUT_DIR = Path("metrics")

MODEL_ORDER = ["GPT-5.4", "Claude Opus 4.7", "Grok 3", "Gemini 2.5 Pro", "DeepSeek R1"]


def mcnemar_test(a, b):
    """
    McNemar's test for paired binary outcomes.
    a, b: boolean Series of same length.
    Returns (statistic, p_value, n_discordant).
    """
    a = a.values.astype(bool)
    b = b.values.astype(bool)
    n10 = ((a == True)  & (b == False)).sum()  # a=yes, b=no
    n01 = ((a == False) & (b == True)).sum()   # a=no,  b=yes
    n_disc = n10 + n01
    if n_disc == 0:
        return None, 1.0, 0
    # exact=False uses chi2 approximation (more appropriate for large n)
    table = [[0, n10], [n01, 0]]
    result = mcnemar(table, exact=False, correction=True)
    return result.statistic, result.pvalue, n_disc


def wilcoxon_test(a, b):
    """Wilcoxon signed-rank test on paired continuous values."""
    diff = a - b
    diff = diff.dropna()
    diff = diff[diff != 0]
    if len(diff) < 10:
        return None, None
    stat, p = wilcoxon(diff, alternative="two-sided")
    return stat, p


def load_rowlevel(path):
    df = pd.read_csv(path)
    df["pc_binary"] = df["premature_closure"].str.strip().str.lower().eq("yes")
    df["score_pct"] = pd.to_numeric(df["score_earned"], errors="coerce") / \
                      pd.to_numeric(df["score_possible"], errors="coerce").replace(0, np.nan)
    return df


def run_analysis(rowlevel_path, dataset_label):
    print(f"\n{'='*60}")
    print(f"{dataset_label}")
    print(f"{'='*60}")

    df = load_rowlevel(rowlevel_path)

    # Filter to valid behavior rows only
    valid_beh = ["ANSWERED", "CLARIFIED", "ESCALATED", "PARTIAL"]
    df = df[df["behavior"].isin(valid_beh)].copy()

    results = []

    # -------------------------------------------------------
    # 1. Baseline vs. Safety (within-model, paired on prompt_id)
    # -------------------------------------------------------
    print("\n--- Baseline vs. Safety (McNemar's test on premature closure) ---")
    for model in MODEL_ORDER:
        base = df[(df["model"] == model) & (df["condition"] == "baseline")].set_index("prompt_id")
        safe = df[(df["model"] == model) & (df["condition"] == "safety")].set_index("prompt_id")
        common = base.index.intersection(safe.index)
        if len(common) < 20:
            continue
        b_pc = base.loc[common, "pc_binary"]
        s_pc = safe.loc[common, "pc_binary"]
        stat, p, n_disc = mcnemar_test(b_pc, s_pc)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        b_rate = b_pc.mean() * 100
        s_rate = s_pc.mean() * 100
        print(f"  {model:<20} baseline={b_rate:.1f}% → safety={s_rate:.1f}%  "
              f"Δ={s_rate-b_rate:+.1f}pp  n_disc={n_disc}  p={p:.4f} {sig}")
        results.append({
            "dataset": dataset_label, "test": "baseline_vs_safety",
            "model_a": model, "model_b": "—", "condition": "baseline→safety",
            "pc_rate_a": round(b_rate, 1), "pc_rate_b": round(s_rate, 1),
            "delta_pp": round(s_rate - b_rate, 1),
            "n_pairs": len(common), "n_discordant": n_disc,
            "statistic": round(stat, 3) if stat else None, "p_value": round(p, 4),
            "significant": sig,
        })

        # Rubric score
        b_sc = base.loc[common, "score_pct"]
        s_sc = safe.loc[common, "score_pct"]
        wstat, wp = wilcoxon_test(b_sc, s_sc)
        wsig = "***" if (wp and wp < 0.001) else ("**" if (wp and wp < 0.01) else ("*" if (wp and wp < 0.05) else "ns"))
        b_sm = b_sc.mean() * 100
        s_sm = s_sc.mean() * 100
        wp_str = f"{wp:.4f}" if wp is not None else "N/A"
        print(f"    rubric score: baseline={b_sm:.1f}% → safety={s_sm:.1f}%  "
              f"Δ={s_sm-b_sm:+.1f}pp  Wilcoxon p={wp_str} {wsig}")
        results.append({
            "dataset": dataset_label, "test": "rubric_baseline_vs_safety",
            "model_a": model, "model_b": "—", "condition": "baseline→safety",
            "pc_rate_a": round(b_sm, 1), "pc_rate_b": round(s_sm, 1),
            "delta_pp": round(s_sm - b_sm, 1),
            "n_pairs": len(common), "n_discordant": None,
            "statistic": round(wstat, 3) if wstat else None,
            "p_value": round(wp, 4) if wp else None,
            "significant": wsig,
        })

    # -------------------------------------------------------
    # 2. Between-model comparisons at baseline (pairwise)
    # -------------------------------------------------------
    print("\n--- Between-model comparisons at baseline (McNemar's, paired on prompt_id) ---")
    baseline_df = df[df["condition"] == "baseline"]
    for m1, m2 in combinations(MODEL_ORDER, 2):
        d1 = baseline_df[baseline_df["model"] == m1].set_index("prompt_id")
        d2 = baseline_df[baseline_df["model"] == m2].set_index("prompt_id")
        common = d1.index.intersection(d2.index)
        if len(common) < 20:
            continue
        pc1 = d1.loc[common, "pc_binary"]
        pc2 = d2.loc[common, "pc_binary"]
        stat, p, n_disc = mcnemar_test(pc1, pc2)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        r1 = pc1.mean() * 100
        r2 = pc2.mean() * 100
        print(f"  {m1:<20} vs {m2:<20}  "
              f"{r1:.1f}% vs {r2:.1f}%  Δ={r2-r1:+.1f}pp  p={p:.4f} {sig}")
        results.append({
            "dataset": dataset_label, "test": "between_model_baseline",
            "model_a": m1, "model_b": m2, "condition": "baseline",
            "pc_rate_a": round(r1, 1), "pc_rate_b": round(r2, 1),
            "delta_pp": round(r2 - r1, 1),
            "n_pairs": len(common), "n_discordant": n_disc,
            "statistic": round(stat, 3) if stat else None, "p_value": round(p, 4),
            "significant": sig,
        })

    return pd.DataFrame(results)


# Run for both datasets
hb_results = run_analysis("metrics/hb_rowlevel.csv", "HealthBench")
rt_results = run_analysis("metrics/hb_redteam_rowlevel.csv", "RedTeam")

all_results = pd.concat([hb_results, rt_results], ignore_index=True)
out_path = OUT_DIR / "hb_significance.csv"
all_results.to_csv(out_path, index=False)
print(f"\n\nSaved → {out_path}")
