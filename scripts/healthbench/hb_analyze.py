"""
HealthBench Analysis — New Pipeline
Reads from results_labeled_healthbench_new/ and results_labeled_healthbench_redteam_new/.
Produces:
  metrics/hb_rowlevel.csv              — one row per (model, condition, question)
  metrics/hb_redteam_rowlevel.csv      — same for red-team
  metrics/hb_aggregate.csv             — aggregate by model × condition × subset
  metrics/hb_aggregate_overall.csv     — aggregate by model × condition (all subsets)
  metrics/hb_redteam_aggregate.csv     — red-team aggregate by model × condition

Run from project root:
  python -u scripts/healthbench/hb_analyze.py
"""

import pandas as pd
from pathlib import Path

# -------------------------------------------------------
# Config
# -------------------------------------------------------
HB_DIR  = Path("results_labeled_healthbench_new")
RT_DIR  = Path("results_labeled_healthbench_redteam_new")
OUT_DIR = Path("metrics")
OUT_DIR.mkdir(exist_ok=True)

MODEL_PRETTY = {
    "gpt54":      "GPT-5.4",
    "claudeopus": "Claude Opus 4.7",
    "grok3":      "Grok 3",
    "gemini":     "Gemini 2.5 Pro",
    "deepseek":   "DeepSeek R1",
}
MODEL_ORDER     = list(MODEL_PRETTY.values())
CONDITION_ORDER = ["baseline", "safety"]

# -------------------------------------------------------
# Helper: parse one labeled CSV → long-format rows
# -------------------------------------------------------
def parse_hb_file(csv_path):
    """
    HB file stem: {model}_{condition}_hb_hb_labeled
    Columns:      prompt_id, question, subset,
                  {prefix}_raw, {prefix}_behavior, {prefix}_premature_closure,
                  {prefix}_score_earned, {prefix}_score_possible, ...
    """
    stem = csv_path.stem  # e.g. gpt54_baseline_hb_hb_labeled
    # strip _labeled
    core = stem.replace("_labeled", "")  # gpt54_baseline_hb_hb
    # strip _hb_hb or _hb suffix to get prefix
    for suf in ["_hb_hb", "_hb"]:
        if core.endswith(suf):
            prefix = core[: -len(suf)]
            break
    else:
        prefix = core

    parts = prefix.split("_", 1)
    model_key = parts[0]
    condition  = parts[1] if len(parts) > 1 else "unknown"

    df = pd.read_csv(csv_path)

    beh_col = f"{prefix}_behavior"
    pc_col  = f"{prefix}_premature_closure"
    se_col  = f"{prefix}_score_earned"
    sp_col  = f"{prefix}_score_possible"

    if beh_col not in df.columns:
        print(f"  ⚠️  {csv_path.name}: missing {beh_col}, skipping")
        return None

    rows = []
    for _, r in df.iterrows():
        behavior = str(r.get(beh_col, "")).strip()
        # Skip rows that are genuine errors (SKIP_ERROR = model API error, not judge error)
        if behavior in ("SKIP_ERROR",):
            continue

        se = r.get(se_col)
        sp = r.get(sp_col)
        score_pct = (float(se) / float(sp)) if pd.notna(se) and pd.notna(sp) and float(sp) > 0 else None

        rows.append({
            "model":              MODEL_PRETTY.get(model_key, model_key),
            "model_key":          model_key,
            "condition":          condition,
            "prompt_id":          r["prompt_id"],
            "subset":             str(r.get("subset", "")).strip(),
            "behavior":           behavior,
            "premature_closure":  str(r.get(pc_col, "")).strip().lower(),
            "score_earned":       se,
            "score_possible":     sp,
            "score_pct":          score_pct,
        })

    return pd.DataFrame(rows)


def parse_rt_file(csv_path):
    """
    Redteam file stem: {model}_{condition}_redteam_hb_labeled
    prefix = {model}_{condition}
    """
    stem = csv_path.stem  # e.g. gpt54_baseline_redteam_hb_labeled
    core = stem.replace("_labeled", "")  # gpt54_baseline_redteam_hb

    for suf in ["_redteam_hb", "_hb"]:
        if core.endswith(suf):
            prefix_base = core[: -len(suf)]  # gpt54_baseline
            break
    else:
        prefix_base = core

    parts = prefix_base.split("_", 1)
    model_key = parts[0]
    condition  = parts[1] if len(parts) > 1 else "unknown"

    df = pd.read_csv(csv_path)

    beh_col = f"{prefix_base}_behavior"
    pc_col  = f"{prefix_base}_premature_closure"
    se_col  = f"{prefix_base}_score_earned"
    sp_col  = f"{prefix_base}_score_possible"

    if beh_col not in df.columns:
        print(f"  ⚠️  {csv_path.name}: missing {beh_col}, skipping")
        return None

    rows = []
    for _, r in df.iterrows():
        behavior = str(r.get(beh_col, "")).strip()
        if behavior in ("SKIP_ERROR",):
            continue

        se = r.get(se_col)
        sp = r.get(sp_col)
        score_pct = (float(se) / float(sp)) if pd.notna(se) and pd.notna(sp) and float(sp) > 0 else None

        rows.append({
            "model":             MODEL_PRETTY.get(model_key, model_key),
            "model_key":         model_key,
            "condition":         condition,
            "prompt_id":         r["prompt_id"],
            "subset":            "red_teaming",
            "behavior":          behavior,
            "premature_closure": str(r.get(pc_col, "")).strip().lower(),
            "score_earned":      se,
            "score_possible":    sp,
            "score_pct":         score_pct,
        })

    return pd.DataFrame(rows)


# -------------------------------------------------------
# Load all HB files
# -------------------------------------------------------
print("Loading HB labeled files...")
hb_frames = []
for f in sorted(HB_DIR.glob("*_labeled.csv")):
    frame = parse_hb_file(f)
    if frame is not None and len(frame) > 0:
        print(f"  {f.name}: {len(frame)} rows (model={frame['model_key'].iloc[0]}, cond={frame['condition'].iloc[0]})")
        hb_frames.append(frame)

hb_df = pd.concat(hb_frames, ignore_index=True)
print(f"\nHB row-level: {len(hb_df)} rows total")
hb_df.to_csv(OUT_DIR / "hb_rowlevel.csv", index=False)
print(f"  Saved → metrics/hb_rowlevel.csv")

# -------------------------------------------------------
# Load all redteam files
# -------------------------------------------------------
print("\nLoading red-team labeled files...")
rt_frames = []
for f in sorted(RT_DIR.glob("*_labeled.csv")):
    frame = parse_rt_file(f)
    if frame is not None and len(frame) > 0:
        model_k = frame["model_key"].iloc[0]
        cond    = frame["condition"].iloc[0]
        print(f"  {f.name}: {len(frame)} rows (model={model_k}, cond={cond})")
        rt_frames.append(frame)

rt_df = pd.concat(rt_frames, ignore_index=True)
print(f"\nRedteam row-level: {len(rt_df)} rows total")
rt_df.to_csv(OUT_DIR / "hb_redteam_rowlevel.csv", index=False)
print(f"  Saved → metrics/hb_redteam_rowlevel.csv")


# -------------------------------------------------------
# Aggregate helper
# -------------------------------------------------------
def aggregate(df, groupby_cols):
    rows = []
    for keys, g in df.groupby(groupby_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groupby_cols, keys))

        n = len(g)
        n_valid_behavior = g["behavior"].isin(["ANSWERED", "CLARIFIED", "ESCALATED", "PARTIAL"]).sum()

        row["N"] = n
        row["N_valid_behavior"] = n_valid_behavior

        # Behavior distribution
        for beh in ["ANSWERED", "CLARIFIED", "ESCALATED", "PARTIAL"]:
            row[f"pct_{beh.lower()}"] = round(g["behavior"].eq(beh).mean() * 100, 1)

        # Premature closure (exclude SKIP_ERROR and PARSE_ERROR rows)
        pc_mask = g["behavior"].isin(["ANSWERED", "CLARIFIED", "ESCALATED", "PARTIAL"])
        pc_sub  = g.loc[pc_mask, "premature_closure"]
        row["premature_closure_rate"] = round(pc_sub.eq("yes").mean() * 100, 1) if len(pc_sub) > 0 else None

        # Rubric score
        score_vals = g["score_pct"].dropna()
        row["mean_rubric_score_pct"] = round(score_vals.mean() * 100, 1) if len(score_vals) > 0 else None

        rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------------------------------
# HB aggregates
# -------------------------------------------------------
print("\n--- HB Aggregate by model × condition × subset ---")
hb_by_subset = aggregate(hb_df, ["model", "condition", "subset"])
hb_by_subset["model"] = pd.Categorical(hb_by_subset["model"], categories=MODEL_ORDER, ordered=True)
hb_by_subset["condition"] = pd.Categorical(hb_by_subset["condition"], categories=CONDITION_ORDER, ordered=True)
hb_by_subset = hb_by_subset.sort_values(["model", "condition", "subset"]).reset_index(drop=True)
hb_by_subset.to_csv(OUT_DIR / "hb_aggregate.csv", index=False)
print(hb_by_subset.to_string(index=False))
print(f"\nSaved → metrics/hb_aggregate.csv")

print("\n--- HB Aggregate by model × condition (overall) ---")
hb_overall = aggregate(hb_df, ["model", "condition"])
hb_overall["model"] = pd.Categorical(hb_overall["model"], categories=MODEL_ORDER, ordered=True)
hb_overall["condition"] = pd.Categorical(hb_overall["condition"], categories=CONDITION_ORDER, ordered=True)
hb_overall = hb_overall.sort_values(["model", "condition"]).reset_index(drop=True)
hb_overall.to_csv(OUT_DIR / "hb_aggregate_overall.csv", index=False)
print(hb_overall.to_string(index=False))
print(f"\nSaved → metrics/hb_aggregate_overall.csv")

# -------------------------------------------------------
# Redteam aggregates
# -------------------------------------------------------
print("\n--- Redteam Aggregate by model × condition ---")
rt_agg = aggregate(rt_df, ["model", "condition"])
rt_agg["model"] = pd.Categorical(rt_agg["model"], categories=MODEL_ORDER, ordered=True)
rt_agg["condition"] = pd.Categorical(rt_agg["condition"], categories=CONDITION_ORDER, ordered=True)
rt_agg = rt_agg.sort_values(["model", "condition"]).reset_index(drop=True)
rt_agg.to_csv(OUT_DIR / "hb_redteam_aggregate.csv", index=False)
print(rt_agg.to_string(index=False))
print(f"\nSaved → metrics/hb_redteam_aggregate.csv")

print("\n✅ Analysis complete.")
