"""
Statistical significance tests for HealthBench results.

For premature closure rate and over-deferral rate (binary):
  - Two-proportion z-test (scipy.stats.proportions_ztest)

For mean rubric score (continuous):
  - Mann-Whitney U test (non-parametric; no normality assumption)

Comparisons run:
  1. Within each model: all 3 pairwise condition comparisons (baseline/safety/clarification)
  2. Across models: all 10 pairwise model comparisons, within each condition

Multiple comparison correction: Benjamini-Hochberg FDR within each comparison family.

Saved to metrics/healthbench_significance.csv
"""

import pandas as pd
import numpy as np
from itertools import combinations
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest
from pathlib import Path

df = pd.read_csv("metrics/healthbench_rowlevel.csv")

df["pc_binary"]     = (df["premature_closure"] == "yes").astype(float)
df["over_deferral"] = (df["behavior"] == "CLARIFIED").astype(float)
df["score_pct"]     = df["score_earned"] / df["score_possible"].replace(0, float("nan"))

MODEL_LABEL     = {"gpt": "GPT-5", "claude": "Claude", "gemini": "Gemini", "llama": "Llama", "deepseek": "DeepSeek"}
CONDITION_LABEL = {"baseline": "Baseline", "safety": "Safety", "clarification": "Clarification-First"}

df["model_label"]     = df["model"].map(MODEL_LABEL)
df["condition_label"] = df["condition"].map(CONDITION_LABEL)

def prop_ztest(a, b):
    """Two-proportion z-test. Returns p-value."""
    count = np.array([a.sum(), b.sum()])
    nobs  = np.array([len(a), len(b)])
    _, p  = proportions_ztest(count, nobs)
    return p

def mannwhitney(a, b):
    """Mann-Whitney U test on non-null values. Returns p-value."""
    a = a.dropna()
    b = b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p

def effect_size_cliff(a, b):
    """Cliff's delta — non-parametric effect size for Mann-Whitney."""
    a = a.dropna().values
    b = b.dropna().values
    n = len(a) * len(b)
    if n == 0:
        return np.nan
    dom = sum(1 if ai > bi else (-1 if ai < bi else 0) for ai in a for bi in b)
    return dom / n

rows = []

# ── 1. Within-model: pairwise condition comparisons ──────────────────────────
conditions = ["Baseline", "Safety", "Clarification-First"]
for model in df["model_label"].unique():
    m = df[df["model_label"] == model]
    pairs = list(combinations(conditions, 2))
    for c1, c2 in pairs:
        g1 = m[m["condition_label"] == c1]
        g2 = m[m["condition_label"] == c2]

        rows.append({
            "family":     "within_model",
            "groupA":     f"{model} / {c1}",
            "groupB":     f"{model} / {c2}",
            "metric":     "premature_closure_rate",
            "rateA":      round(g1["pc_binary"].mean(), 3),
            "rateB":      round(g2["pc_binary"].mean(), 3),
            "nA":         len(g1),
            "nB":         len(g2),
            "p_raw":      prop_ztest(g1["pc_binary"], g2["pc_binary"]),
            "effect":     round(g1["pc_binary"].mean() - g2["pc_binary"].mean(), 3),
        })
        rows.append({
            "family":     "within_model",
            "groupA":     f"{model} / {c1}",
            "groupB":     f"{model} / {c2}",
            "metric":     "over_deferral_rate",
            "rateA":      round(g1["over_deferral"].mean(), 3),
            "rateB":      round(g2["over_deferral"].mean(), 3),
            "nA":         len(g1),
            "nB":         len(g2),
            "p_raw":      prop_ztest(g1["over_deferral"], g2["over_deferral"]),
            "effect":     round(g1["over_deferral"].mean() - g2["over_deferral"].mean(), 3),
        })
        rows.append({
            "family":     "within_model",
            "groupA":     f"{model} / {c1}",
            "groupB":     f"{model} / {c2}",
            "metric":     "mean_rubric_score",
            "rateA":      round(g1["score_pct"].mean(), 3),
            "rateB":      round(g2["score_pct"].mean(), 3),
            "nA":         len(g1),
            "nB":         len(g2),
            "p_raw":      mannwhitney(g1["score_pct"], g2["score_pct"]),
            "effect":     round(effect_size_cliff(g1["score_pct"], g2["score_pct"]), 3),
        })

# ── 2. Across-model: pairwise model comparisons within each condition ────────
models = list(df["model_label"].unique())
for condition in conditions:
    c = df[df["condition_label"] == condition]
    for m1, m2 in combinations(models, 2):
        g1 = c[c["model_label"] == m1]
        g2 = c[c["model_label"] == m2]

        rows.append({
            "family":     "across_model",
            "groupA":     f"{m1} / {condition}",
            "groupB":     f"{m2} / {condition}",
            "metric":     "premature_closure_rate",
            "rateA":      round(g1["pc_binary"].mean(), 3),
            "rateB":      round(g2["pc_binary"].mean(), 3),
            "nA":         len(g1),
            "nB":         len(g2),
            "p_raw":      prop_ztest(g1["pc_binary"], g2["pc_binary"]),
            "effect":     round(g1["pc_binary"].mean() - g2["pc_binary"].mean(), 3),
        })
        rows.append({
            "family":     "across_model",
            "groupA":     f"{m1} / {condition}",
            "groupB":     f"{m2} / {condition}",
            "metric":     "mean_rubric_score",
            "rateA":      round(g1["score_pct"].mean(), 3),
            "rateB":      round(g2["score_pct"].mean(), 3),
            "nA":         len(g1),
            "nB":         len(g2),
            "p_raw":      mannwhitney(g1["score_pct"], g2["score_pct"]),
            "effect":     round(effect_size_cliff(g1["score_pct"], g2["score_pct"]), 3),
        })

result = pd.DataFrame(rows)

# ── BH FDR correction within each family × metric ───────────────────────────
result["p_fdr"] = np.nan
result["significant"] = False

for (family, metric), grp in result.groupby(["family", "metric"]):
    idx = grp.index
    p_vals = grp["p_raw"].values
    valid  = ~np.isnan(p_vals)
    if valid.sum() == 0:
        continue
    corrected = np.full(len(p_vals), np.nan)
    _, p_adj, _, _ = multipletests(p_vals[valid], method="fdr_bh")
    corrected[valid] = p_adj
    result.loc[idx, "p_fdr"] = corrected
    result.loc[idx, "significant"] = corrected < 0.05

result["p_raw"] = result["p_raw"].round(4)
result["p_fdr"] = result["p_fdr"].round(4)

out_path = Path("metrics/healthbench_significance.csv")
result.to_csv(out_path, index=False)
print(f"Saved → {out_path} ({len(result)} comparisons)")
print()

# ── Summary: significant findings ───────────────────────────────────────────
sig = result[result["significant"] == True].copy()
print(f"Significant comparisons (FDR < 0.05): {len(sig)} / {len(result)}")
print()

for metric in ["premature_closure_rate", "over_deferral_rate", "mean_rubric_score"]:
    sub = sig[sig["metric"] == metric]
    if len(sub) == 0:
        continue
    print(f"── {metric} ({len(sub)} significant) ──")
    for _, r in sub.iterrows():
        direction = ">" if r["rateA"] > r["rateB"] else "<"
        print(f"  {r['groupA']} {direction} {r['groupB']}  "
              f"(Δ={r['effect']:+.3f}, p_fdr={r['p_fdr']:.4f})")
    print()
