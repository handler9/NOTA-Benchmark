"""
Judge Agreement Analysis — GPT judge vs. Claude judge
Computes percent agreement and Cohen's kappa for:
  - behavior label (4-class)
  - premature_closure (binary yes/no)
  - rubric score (correlation)

Covers both HB (861q) and red-team (191q).

Run from project root:
  python -u scripts/healthbench/hb_judge_agreement.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

GPT_HB_DIR   = Path("results_labeled_healthbench_new")
CLA_HB_DIR   = Path("results_labeled_healthbench_claude_new")
GPT_RT_DIR   = Path("results_labeled_healthbench_redteam_new")
CLA_RT_DIR   = Path("results_labeled_healthbench_redteam_claude_new")
OUT_DIR      = Path("metrics")

MODEL_PRETTY = {
    "gpt54":      "GPT-5.4",
    "claudeopus": "Claude Opus 4.7",
    "grok3":      "Grok 3",
    "gemini":     "Gemini 2.5 Pro",
    "deepseek":   "DeepSeek R1",
}
MODEL_ORDER = list(MODEL_PRETTY.values())

VALID_BEHAVIORS = {"ANSWERED", "CLARIFIED", "ESCALATED", "PARTIAL"}


def get_prefix(stem):
    """Extract model_condition prefix from labeled file stem."""
    core = stem.replace("_labeled", "")
    for suf in ["_hb_hb", "_redteam_hb", "_hb"]:
        if core.endswith(suf):
            return core[: -len(suf)]
    return core


def compute_agreement(gpt_dir, cla_dir, dataset_label):
    rows = []
    files = sorted(gpt_dir.glob("*_labeled.csv"))

    for gpt_path in files:
        cla_path = cla_dir / gpt_path.name
        if not cla_path.exists():
            print(f"  ⚠️  Claude judge missing: {gpt_path.name}")
            continue

        prefix = get_prefix(gpt_path.stem)
        parts = prefix.split("_", 1)
        model_key = parts[0]
        condition  = parts[1] if len(parts) > 1 else "unknown"

        gpt_df = pd.read_csv(gpt_path)
        cla_df = pd.read_csv(cla_path)

        beh_col = f"{prefix}_behavior"
        pc_col  = f"{prefix}_premature_closure"
        se_col  = f"{prefix}_score_earned"
        sp_col  = f"{prefix}_score_possible"

        # Merge on prompt_id
        merged = gpt_df[["prompt_id", beh_col, pc_col, se_col, sp_col]].merge(
            cla_df[["prompt_id", beh_col, pc_col, se_col, sp_col]],
            on="prompt_id",
            suffixes=("_gpt", "_cla"),
        )

        # Filter to rows where BOTH judges gave a valid behavior
        beh_gpt = merged[f"{beh_col}_gpt"].str.strip()
        beh_cla = merged[f"{beh_col}_cla"].str.strip()
        valid_mask = beh_gpt.isin(VALID_BEHAVIORS) & beh_cla.isin(VALID_BEHAVIORS)
        m = merged[valid_mask].copy()

        if len(m) == 0:
            print(f"  ⚠️  No valid rows for {gpt_path.name}")
            continue

        beh_g = m[f"{beh_col}_gpt"].str.strip()
        beh_c = m[f"{beh_col}_cla"].str.strip()

        # -- Behavior agreement --
        beh_agree = (beh_g == beh_c).mean()
        try:
            beh_kappa = cohen_kappa_score(beh_g, beh_c)
        except Exception:
            beh_kappa = None

        # -- Premature closure agreement (binary) --
        pc_g = m[f"{pc_col}_gpt"].str.strip().str.lower()
        pc_c = m[f"{pc_col}_cla"].str.strip().str.lower()
        pc_valid = pc_g.isin(["yes", "no"]) & pc_c.isin(["yes", "no"])
        pc_m_g = pc_g[pc_valid]
        pc_m_c = pc_c[pc_valid]

        pc_agree = (pc_m_g == pc_m_c).mean() if len(pc_m_g) > 0 else None
        try:
            pc_kappa = cohen_kappa_score(pc_m_g, pc_m_c) if len(pc_m_g) > 0 else None
        except Exception:
            pc_kappa = None

        # -- Rubric score correlation --
        se_g = pd.to_numeric(m[f"{se_col}_gpt"], errors="coerce")
        se_c = pd.to_numeric(m[f"{se_col}_cla"], errors="coerce")
        sp_g = pd.to_numeric(m[f"{sp_col}_gpt"], errors="coerce")
        sp_c = pd.to_numeric(m[f"{sp_col}_cla"], errors="coerce")

        score_g = (se_g / sp_g.replace(0, np.nan))
        score_c = (se_c / sp_c.replace(0, np.nan))
        sc_valid = score_g.notna() & score_c.notna()

        score_corr = score_g[sc_valid].corr(score_c[sc_valid]) if sc_valid.sum() > 1 else None

        # -- PC rate per judge (for reference) --
        pc_rate_gpt = pc_m_g.eq("yes").mean() if len(pc_m_g) > 0 else None
        pc_rate_cla = pc_m_c.eq("yes").mean() if len(pc_m_c) > 0 else None

        rows.append({
            "dataset":           dataset_label,
            "model":             MODEL_PRETTY.get(model_key, model_key),
            "condition":         condition,
            "N_valid":           int(valid_mask.sum()),
            "behavior_agree":    round(beh_agree * 100, 1),
            "behavior_kappa":    round(beh_kappa, 3) if beh_kappa is not None else None,
            "pc_agree":          round(pc_agree * 100, 1) if pc_agree is not None else None,
            "pc_kappa":          round(pc_kappa, 3) if pc_kappa is not None else None,
            "pc_rate_gpt":       round(pc_rate_gpt * 100, 1) if pc_rate_gpt is not None else None,
            "pc_rate_claude":    round(pc_rate_cla * 100, 1) if pc_rate_cla is not None else None,
            "score_correlation": round(score_corr, 3) if score_corr is not None else None,
        })

        print(f"  {gpt_path.name}: N={valid_mask.sum()} | "
              f"beh_agree={beh_agree*100:.1f}% κ={beh_kappa:.3f} | "
              f"pc_agree={pc_agree*100:.1f}% κ={pc_kappa:.3f} | "
              f"score_r={score_corr:.3f}" if score_corr else
              f"  {gpt_path.name}: N={valid_mask.sum()} | "
              f"beh_agree={beh_agree*100:.1f}%")

    return pd.DataFrame(rows)


print("=" * 60)
print("HB (861q) Judge Agreement")
print("=" * 60)
hb_agree = compute_agreement(GPT_HB_DIR, CLA_HB_DIR, "HealthBench")

print("\n" + "=" * 60)
print("Red-team (191q) Judge Agreement")
print("=" * 60)
rt_agree = compute_agreement(GPT_RT_DIR, CLA_RT_DIR, "RedTeam")

all_agree = pd.concat([hb_agree, rt_agree], ignore_index=True)
all_agree["model"] = pd.Categorical(all_agree["model"], categories=MODEL_ORDER, ordered=True)
all_agree = all_agree.sort_values(["dataset", "model", "condition"]).reset_index(drop=True)

out_path = OUT_DIR / "hb_judge_agreement.csv"
all_agree.to_csv(out_path, index=False)

print("\n\n--- Summary ---")
print(all_agree.to_string(index=False))
print(f"\nSaved → {out_path}")

# -- Overall summary stats --
print("\n--- Overall (mean across all files) ---")
for ds in ["HealthBench", "RedTeam"]:
    sub = all_agree[all_agree["dataset"] == ds]
    print(f"\n{ds}:")
    print(f"  Behavior agreement:     {sub['behavior_agree'].mean():.1f}%  (κ = {sub['behavior_kappa'].mean():.3f})")
    print(f"  Premature closure agr:  {sub['pc_agree'].mean():.1f}%  (κ = {sub['pc_kappa'].mean():.3f})")
    if sub["score_correlation"].notna().any():
        print(f"  Rubric score r:         {sub['score_correlation'].mean():.3f}")
