from pathlib import Path
import pandas as pd
import numpy as np

# Use a non-interactive backend (prevents GUI/backend issues in VS Code/remote runs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Robust repo-root detection
# - Walk upward from this script until we find a folder
#   that contains "metrics/".
# ------------------------------------------------------
ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "metrics").exists():
    ROOT = ROOT.parent

# Optional: if metrics still not found, fall back to two-level-up guess
if not (ROOT / "metrics").exists():
    ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "metrics" / "medqa_results.csv"
RAW_DIR = ROOT / "results_raw"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True, parents=True)

# -------------------------
# Helpers
# -------------------------
def pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def safe_lower(x):
    return str(x).strip().lower()

def parse_model_prompt_from_filename(path: Path):
    """
    Expects filenames like:
      gemini_baseline.csv
      deepseek_doublecheck.csv
      gpt_baseline.csv
      claude_think.csv
    Returns (model, prompt) best-effort.
    """
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "unknown"

def detect_raw_cols(df):
    """
    Your raw CSVs look like:
      <model>_choice, <model>_abstain_code, <model>_confidence
    Detect them automatically.
    """
    choice = [c for c in df.columns if c.endswith("_choice")]
    abst  = [c for c in df.columns if c.endswith("_abstain_code")]
    conf  = [c for c in df.columns if c.endswith("_confidence")]

    choice_col = choice[0] if choice else None
    abst_col   = abst[0] if abst else None
    conf_col   = conf[0] if conf else None

    model = None
    if choice_col:
        model = choice_col.replace("_choice", "")
    return model, choice_col, abst_col, conf_col

# -------------------------
# Load metrics (aggregated)
# -------------------------
if not METRICS.exists():
    raise FileNotFoundError(
        f"Missing metrics file: {METRICS}\n"
        f"Detected ROOT as: {ROOT}\n"
        f"Tip: verify your repo has {ROOT/'metrics'} and that the CSV filename matches."
    )

dfm = pd.read_csv(METRICS)

col_model   = pick_col(dfm, ["model", "model_name"])
col_prompt  = pick_col(dfm, ["prompt", "prompt_name", "setting"])
col_acc     = pick_col(dfm, ["intact_accuracy", "accuracy_intact", "acc_intact"])
col_far     = pick_col(dfm, ["false_action_rate_true_nota", "true_nota_false_action_rate", "far_true_nota"])
col_overall = pick_col(dfm, ["overall_accuracy", "accuracy_overall", "accuracy"])

need = [("model", col_model), ("prompt", col_prompt), ("intact_accuracy", col_acc), ("true_nota_far", col_far)]
missing = [n for n, c in need if c is None]
if missing:
    raise ValueError(
        f"Missing columns in {METRICS} for: {missing}\n"
        f"Columns present: {list(dfm.columns)}"
    )

# -------------------------
# Compute TRUE-NOTA abstention rate from FALSE ACTION RATE
# FAR = chose A/B/C/D on TRUE-NOTA items
# Abstention = 1 - FAR
# -------------------------
dfm["true_nota_abstention_rate"] = 1.0 - pd.to_numeric(dfm[col_far], errors="coerce")
col_abst = "true_nota_abstention_rate"

# Normalize prompt labels a bit (helps ordering)
dfm["_prompt_l"] = dfm[col_prompt].astype(str).map(safe_lower)

def prompt_sort_key(p):
    pl = safe_lower(p).replace("-", " ").replace("_", " ")
    pl = pl.replace("think then decide", "think")
    pl = pl.replace("think-then-decide", "think")
    pl = pl.replace("thinkthen", "think")
    pl = pl.replace("double-check", "doublecheck")
    pl = pl.replace("double check", "doublecheck")
    if pl in ["baseline", "safety", "doublecheck", "think"]:
        order = {"baseline": 0, "safety": 1, "doublecheck": 2, "think": 3}
        return order[pl]
    return 999

prompts_sorted = sorted(dfm[col_prompt].astype(str).unique(), key=prompt_sort_key)
models_sorted = sorted(dfm[col_model].astype(str).unique())

# -------------------------
# FIG 1: Overall grouped bars (baseline prompt)
# (INTACT accuracy vs TRUE-NOTA abstention)
# -------------------------
PROMPT_TO_PLOT = "baseline"
sub = dfm[dfm["_prompt_l"].eq(PROMPT_TO_PLOT)].copy()
if sub.empty:
    first = dfm[col_prompt].astype(str).iloc[0]
    sub = dfm[dfm[col_prompt].astype(str).eq(first)].copy()
    PROMPT_TO_PLOT = safe_lower(first)

sub = sub.sort_values(col_model)
x = range(len(sub))
w = 0.4

plt.figure(figsize=(10, 5))
plt.bar([i - w/2 for i in x], sub[col_acc], w, label="INTACT accuracy")
plt.bar([i + w/2 for i in x], sub[col_abst], w, label="TRUE-NOTA abstention rate")
plt.xticks(list(x), sub[col_model], rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("Rate")
plt.title(f"500Q overall performance (prompt = {PROMPT_TO_PLOT})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_overall.pdf")
plt.close()

# -------------------------
# FIG 2: Prompt sensitivity (TRUE-NOTA abstention across prompts)
# -------------------------
plt.figure(figsize=(10, 5))
for model, g in dfm.groupby(col_model):
    g = g.copy()
    g = g.set_index(col_prompt).reindex(prompts_sorted).reset_index()
    plt.plot(g[col_prompt].astype(str), g[col_abst], marker="o", label=str(model))
plt.ylim(0, 1)
plt.ylabel("TRUE-NOTA abstention rate")
plt.xlabel("Prompt")
plt.xticks(rotation=30, ha="right")
plt.title("Prompt Effect on Abstention Behavior (MedQA)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_prompt_effect.pdf")
plt.close()

# -------------------------
# FIG 3: Accuracy–Abstention tradeoff scatter (model × prompt)
# -------------------------
plt.figure(figsize=(7, 6))
for _, r in dfm.iterrows():
    if pd.isna(r[col_acc]) or pd.isna(r[col_abst]):
        continue
    plt.scatter(r[col_acc], r[col_abst])
    plt.text(r[col_acc], r[col_abst], f"{r[col_model]}|{r[col_prompt]}", fontsize=7)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("INTACT accuracy")
plt.ylabel("TRUE-NOTA abstention rate")
plt.title("Accuracy–Abstention tradeoff (model × prompt)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_tradeoff.pdf")
plt.close()

# -------------------------
# FIG 4: Heatmap of TRUE-NOTA abstention (model × prompt)
# -------------------------
pivot_abst = dfm.pivot_table(index=col_model, columns=col_prompt, values=col_abst, aggfunc="mean")
pivot_abst = pivot_abst.reindex(index=models_sorted, columns=prompts_sorted)

plt.figure(figsize=(max(8, len(prompts_sorted)*1.3), max(4, len(models_sorted)*0.7)))
plt.imshow(pivot_abst.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(pivot_abst.columns)), pivot_abst.columns.astype(str), rotation=30, ha="right")
plt.yticks(range(len(pivot_abst.index)), pivot_abst.index.astype(str))
plt.colorbar(label="TRUE-NOTA abstention rate")
plt.title("Heatmap: TRUE-NOTA abstention rate (model × prompt)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig4_heatmap_abstention.pdf")
plt.close()

# -------------------------
# FIG 5: Heatmap of INTACT accuracy (model × prompt)
# -------------------------
pivot_acc = dfm.pivot_table(index=col_model, columns=col_prompt, values=col_acc, aggfunc="mean")
pivot_acc = pivot_acc.reindex(index=models_sorted, columns=prompts_sorted)

plt.figure(figsize=(max(8, len(prompts_sorted)*1.3), max(4, len(models_sorted)*0.7)))
plt.imshow(pivot_acc.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(pivot_acc.columns)), pivot_acc.columns.astype(str), rotation=30, ha="right")
plt.yticks(range(len(pivot_acc.index)), pivot_acc.index.astype(str))
plt.colorbar(label="INTACT accuracy")
plt.title("Heatmap: INTACT accuracy (model × prompt)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig5_heatmap_accuracy.pdf")
plt.close()

# -------------------------
# FIG 6: Delta abstention vs baseline (per model, per prompt)
# -------------------------
baseline_map = (
    dfm[dfm["_prompt_l"].eq("baseline")]
    .set_index(col_model)[col_abst]
    .to_dict()
)

df_delta = dfm.copy()
df_delta["baseline_abst"] = df_delta[col_model].map(baseline_map)
df_delta["delta_abst_vs_baseline"] = df_delta[col_abst] - df_delta["baseline_abst"]

df_delta = df_delta[~df_delta["baseline_abst"].isna()].copy()
df_delta = df_delta.sort_values(["delta_abst_vs_baseline"])

plt.figure(figsize=(10, 6))
plt.scatter(range(len(df_delta)), df_delta["delta_abst_vs_baseline"])
plt.axhline(0, linewidth=1)
plt.xticks(
    range(len(df_delta)),
    [f"{m}|{p}" for m, p in zip(df_delta[col_model].astype(str), df_delta[col_prompt].astype(str))],
    rotation=60,
    ha="right",
    fontsize=7,
)
plt.ylabel("Δ ABSTENTION RATE (TRUE-NOTA) vs baseline")
plt.title("Change in abstention vs baseline (higher is better)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig6_delta_abst_vs_baseline.pdf")
plt.close()

# -------------------------
# Load raw per-question outputs (results_raw/*.csv)
# -------------------------
raw_files = sorted(RAW_DIR.glob("*.csv"))
raw_all = []

if raw_files:
    for f in raw_files:
        rdf = pd.read_csv(f)
        fname_model, fname_prompt = parse_model_prompt_from_filename(f)
        model_prefix, c_choice, c_abst, c_conf = detect_raw_cols(rdf)

        model = model_prefix if model_prefix else fname_model
        prompt = fname_prompt

        if "question_id" not in rdf.columns or c_choice is None:
            continue

        tmp = pd.DataFrame({
            "question_id": rdf["question_id"],
            "model": model,
            "prompt": prompt,
            "choice": rdf[c_choice],
            "abstain_code": rdf[c_abst] if (c_abst is not None and c_abst in rdf.columns) else None,
            "confidence": rdf[c_conf] if (c_conf is not None and c_conf in rdf.columns) else None,
        })
        raw_all.append(tmp)

raw = pd.concat(raw_all, ignore_index=True) if raw_all else None

# -------------------------
# Raw-based figs (trend hunting)
# -------------------------
if raw is not None and len(raw) > 0:
    raw["prompt_l"] = raw["prompt"].astype(str).map(safe_lower)
    raw["picked_option"] = raw["choice"].isin(["A", "B", "C", "D"])

    # FIG 7: Confidence distributions by prompt (boxplot per model)
    for model, g in raw.groupby("model"):
        g = g.copy()
        g["confidence"] = pd.to_numeric(g["confidence"], errors="coerce")
        g = g.dropna(subset=["confidence"])
        if g.empty:
            continue

        pr = sorted(g["prompt"].unique(), key=prompt_sort_key)
        data = [g[g["prompt"] == p]["confidence"].values for p in pr]

        plt.figure(figsize=(max(8, len(pr)*1.2), 5))
        plt.boxplot(data, labels=[str(p) for p in pr], showfliers=False)
        plt.ylim(0, 1)
        plt.ylabel("Confidence")
        plt.title(f"Confidence distribution by prompt — {model}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(OUTDIR / f"fig7_confidence_box_{model}.pdf")
        plt.close()

    # FIG 8: Abstain-code breakdown by (model × prompt) (stacked bars)
    raw["abstain_code"] = raw["abstain_code"].astype(str).replace({"nan": ""})
    raw["abstain_code_norm"] = raw["abstain_code"].map(lambda x: x.strip().upper() if x else "")

    abst = raw[~raw["picked_option"]].copy()
    if not abst.empty:
        tab = (
            abst.groupby(["model", "prompt", "abstain_code_norm"])
            .size()
            .rename("n")
            .reset_index()
        )
        totals = tab.groupby(["model", "prompt"])["n"].sum().rename("total").reset_index()
        tab = tab.merge(totals, on=["model", "prompt"], how="left")
        tab["share"] = tab["n"] / tab["total"]

        pivot = tab.pivot_table(
            index=["model", "prompt"],
            columns="abstain_code_norm",
            values="share",
            aggfunc="sum"
        ).fillna(0)

        if "" in pivot.columns:
            pivot = pivot.drop(columns=[""])

        labels = [f"{m}|{p}" for m, p in pivot.index]
        bottoms = np.zeros(len(pivot))
        plt.figure(figsize=(max(10, len(labels)*0.35), 6))
        for code in pivot.columns:
            vals = pivot[code].values
            plt.bar(range(len(vals)), vals, bottom=bottoms, label=str(code))
            bottoms += vals

        plt.ylim(0, 1)
        plt.ylabel("Share of abstentions")
        plt.title("Abstain-code composition (by model × prompt) — among abstained questions")
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=7)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(OUTDIR / "fig8_abstain_code_stacked.pdf")
        plt.close()

    # FIG 9: Choice-share (A/B/C/D vs abstain) by (model × prompt)
    raw["choice_cat"] = np.where(raw["picked_option"], raw["choice"].astype(str), "ABSTAIN")
    tab2 = (
        raw.groupby(["model", "prompt", "choice_cat"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals2 = tab2.groupby(["model", "prompt"])["n"].sum().rename("total").reset_index()
    tab2 = tab2.merge(totals2, on=["model", "prompt"], how="left")
    tab2["share"] = tab2["n"] / tab2["total"]

    pivot2 = tab2.pivot_table(
        index=["model", "prompt"],
        columns="choice_cat",
        values="share",
        aggfunc="sum"
    ).fillna(0)

    cols_order = [c for c in ["A", "B", "C", "D", "ABSTAIN"] if c in pivot2.columns] + \
                 [c for c in pivot2.columns if c not in ["A", "B", "C", "D", "ABSTAIN"]]
    pivot2 = pivot2[cols_order]

    labels2 = [f"{m}|{p}" for m, p in pivot2.index]
    bottoms = np.zeros(len(pivot2))
    plt.figure(figsize=(max(10, len(labels2)*0.35), 6))
    for cat in pivot2.columns:
        vals = pivot2[cat].values
        plt.bar(range(len(vals)), vals, bottom=bottoms, label=str(cat))
        bottoms += vals

    plt.ylim(0, 1)
    plt.ylabel("Share of questions")
    plt.title("Choice distribution (A/B/C/D vs abstain) by model × prompt")
    plt.xticks(range(len(labels2)), labels2, rotation=60, ha="right", fontsize=7)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig9_choice_share_stacked.pdf")
    plt.close()

    # FIG 10: Prompt stability (within-model consistency across prompts)
    stability_rows = []
    for model, g in raw.groupby("model"):
        wide = g.pivot_table(index="question_id", columns="prompt", values="choice", aggfunc="first")
        norm = wide.applymap(lambda x: x if x in ["A", "B", "C", "D"] else np.nan)

        def stable_row(row):
            vals = row.dropna().values
            if len(vals) <= 1:
                return np.nan
            return float(len(set(vals)) == 1)

        stable = norm.apply(stable_row, axis=1)
        stability = stable.mean(skipna=True)
        n_used = stable.notna().sum()
        stability_rows.append({"model": model, "choice_stability": stability, "n_questions_used": n_used})

    stab = pd.DataFrame(stability_rows).sort_values("choice_stability", ascending=False)

    plt.figure(figsize=(8, 4))
    plt.bar(stab["model"].astype(str), stab["choice_stability"])
    plt.ylim(0, 1)
    plt.ylabel("Share of questions with same A/B/C/D across prompts")
    plt.title("Within-model prompt stability (A/B/C/D only)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig10_prompt_stability.pdf")
    plt.close()

print(f"\n✅ Saved figures to: {OUTDIR}\n")