"""
make_all_figures_500q.py

Generates paper-ready figures from:
- metrics/500q_results2.csv (aggregated model × prompt metrics)
- results_raw/*.csv (per-question raw outputs; e.g., claude_baseline.csv)

Outputs: figures/*.pdf
"""

from pathlib import Path
import pandas as pd
import numpy as np

# Non-interactive backend (safe for VS Code/debug, headless runs)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------------------------------------
# Robust repo-root detection: walk upward until /metrics exists
# ------------------------------------------------------
ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "metrics").exists():
    ROOT = ROOT.parent

if not (ROOT / "metrics").exists():
    # fallback guess if script is nested unusually
    ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "metrics" / "500q_results2.csv"
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
      llama_safety.csv

    Returns (model_stub, prompt_stub) best-effort.
    """
    stem = path.stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "unknown"

def detect_raw_cols(df):
    """
    Detect raw columns like:
      <model>_choice, <model>_abstain_code, <model>_confidence
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

def normalize_model_name(m):
    s = str(m).strip()
    sl = s.lower()
    if "gemini" in sl:
        return "Gemini"
    if sl.startswith("gpt") or "gpt" in sl:
        # keep GPT-5 label if you want; otherwise "GPT"
        return "GPT-5" if "5" in sl else "GPT"
    if "claude" in sl:
        return "Claude"
    if "deepseek" in sl:
        return "DeepSeek"
    if "llama" in sl:
        return "Llama"
    return s

def normalize_prompt_name(p):
    """
    Normalizes raw+metrics prompt variants into canonical labels used in plots.
    """
    pl = safe_lower(p).replace("-", "").replace("_", "").replace(" ", "")
    # canonical:
    if pl in ["baseline"]:
        return "baseline"
    if pl in ["safety", "safetyprompt"]:
        return "safety-prompt"
    if pl in ["doublecheck", "answerthendoublecheck", "answerthendoublecheckprompt"]:
        return "answer-then-double-check"
    if pl in ["think", "thinkthendecide", "thinkthendecideprompt"]:
        return "think-then-decide"
    # fallback: keep original
    return str(p)

def prompt_sort_key(p):
    # desired order across all figures
    order = {
        "baseline": 0,
        "safety-prompt": 1,
        "answer-then-double-check": 2,
        "think-then-decide": 3,
    }
    return order.get(normalize_prompt_name(p), 999)

def to_rate(x):
    """Coerce values like '92%', '0.92', '92' into [0,1] float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if v > 1.0:
        return v / 100.0
    return v


# -------------------------
# Load metrics (aggregated)
# -------------------------
if not METRICS.exists():
    raise FileNotFoundError(f"Missing metrics file: {METRICS}")

dfm = pd.read_csv(METRICS)

col_model  = pick_col(dfm, ["model", "model_name"])
col_prompt = pick_col(dfm, ["prompt", "prompt_name", "setting"])
col_acc    = pick_col(dfm, ["intact_accuracy", "accuracy_intact", "acc_intact"])
col_far    = pick_col(dfm, ["false_action_rate_true_nota", "true_nota_false_action_rate", "far_true_nota"])
col_overall = pick_col(dfm, ["overall_accuracy", "accuracy_overall", "accuracy"])

# NEW: abstention rate on TRUE-NOTA for prompt-effect plot
col_abst_true_nota = pick_col(dfm, ["abstain_rate_true_nota"])

need = [
    ("model", col_model),
    ("prompt", col_prompt),
    ("intact_accuracy", col_acc),
    ("true_nota_far", col_far),
]
missing = [n for n, c in need if c is None]
if missing:
    raise ValueError(
        f"Missing columns in {METRICS} for: {missing}\n"
        f"Columns present: {list(dfm.columns)}"
    )

if col_abst_true_nota is None:
    raise ValueError(
        "Missing abstention column in metrics file. Expected: 'abstain_rate_true_nota'.\n"
        f"Columns present: {list(dfm.columns)}"
    )

# Normalize model/prompt labels and coerce rates
dfm[col_model] = dfm[col_model].map(normalize_model_name)
dfm[col_prompt] = dfm[col_prompt].map(normalize_prompt_name)

dfm[col_acc] = dfm[col_acc].map(to_rate)
dfm[col_far] = dfm[col_far].map(to_rate)
dfm[col_abst_true_nota] = dfm[col_abst_true_nota].map(to_rate)

dfm["_prompt_l"] = dfm[col_prompt].astype(str).map(safe_lower)

prompts_sorted = sorted(dfm[col_prompt].astype(str).unique(), key=prompt_sort_key)
models_sorted = sorted(dfm[col_model].astype(str).unique())


# -------------------------
# FIG 1: Overall grouped bars (baseline prompt)
# -------------------------
PROMPT_TO_PLOT = "baseline"
sub = dfm[dfm[col_prompt].astype(str).eq(PROMPT_TO_PLOT)].copy()
if sub.empty:
    # fallback to first prompt present
    first = dfm[col_prompt].astype(str).iloc[0]
    sub = dfm[dfm[col_prompt].astype(str).eq(first)].copy()
    PROMPT_TO_PLOT = first

sub = sub.sort_values(col_model)
x = range(len(sub))
w = 0.4

plt.figure(figsize=(10, 5))
plt.bar([i - w/2 for i in x], sub[col_acc], w, label="INTACT accuracy")
plt.bar([i + w/2 for i in x], sub[col_far], w, label="TRUE-NOTA false action rate")
plt.xticks(list(x), sub[col_model], rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("Rate")
plt.title(f"500Q overall performance (prompt = {PROMPT_TO_PLOT})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_overall.pdf")
plt.close()


# -------------------------
# FIG 2: Prompt effect (ABSTENTION rate across prompts) — NOW IN PERCENT
# -------------------------
plt.figure(figsize=(10, 5))
for model, g in dfm.groupby(col_model):
    g = g.copy()
    g = g.set_index(col_prompt).reindex(prompts_sorted).reset_index()
    plt.plot(
        g[col_prompt].astype(str),
        g[col_abst_true_nota] * 100.0,
        marker="o",
        label=str(model),
    )
plt.ylim(0, 100)
plt.ylabel("Abstention rate on TRUE-NOTA (%)")
plt.xlabel("Prompt")
plt.xticks(rotation=30, ha="right")
plt.title("Prompt effect on abstention behavior (500Q)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_prompt_effect.pdf")
plt.close()


# -------------------------
# FIG 3: Accuracy–Safety tradeoff scatter (model × prompt)
# -------------------------
plt.figure(figsize=(7, 6))
for _, r in dfm.iterrows():
    plt.scatter(r[col_acc], r[col_far])
    plt.text(r[col_acc], r[col_far], f"{r[col_model]}|{r[col_prompt]}", fontsize=7)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("INTACT accuracy")
plt.ylabel("TRUE-NOTA false action rate")
plt.title("Accuracy–Safety tradeoff (model × prompt)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_tradeoff.pdf")
plt.close()


# -------------------------
# FIG 4: Heatmap of TRUE-NOTA FAR (model × prompt)
# -------------------------
pivot_far = dfm.pivot_table(index=col_model, columns=col_prompt, values=col_far, aggfunc="mean")
pivot_far = pivot_far.reindex(index=models_sorted, columns=prompts_sorted)

plt.figure(figsize=(max(8, len(prompts_sorted)*1.3), max(4, len(models_sorted)*0.7)))
plt.imshow(pivot_far.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(pivot_far.columns)), pivot_far.columns.astype(str), rotation=30, ha="right")
plt.yticks(range(len(pivot_far.index)), pivot_far.index.astype(str))
plt.colorbar(label="TRUE-NOTA false action rate")
plt.title("Heatmap: TRUE-NOTA false action rate (model × prompt)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig4_heatmap_far.pdf")
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
# FIG 6: Delta FAR vs baseline (per model, per prompt)
# -------------------------
baseline_map = (
    dfm[dfm[col_prompt].astype(str).eq("baseline")]
    .set_index(col_model)[col_far]
    .to_dict()
)

df_delta = dfm.copy()
df_delta["baseline_far"] = df_delta[col_model].map(baseline_map)
df_delta["delta_far_vs_baseline"] = df_delta[col_far] - df_delta["baseline_far"]
df_delta = df_delta[~df_delta["baseline_far"].isna()].copy()
df_delta = df_delta.sort_values(["delta_far_vs_baseline"])

plt.figure(figsize=(10, 6))
plt.scatter(range(len(df_delta)), df_delta["delta_far_vs_baseline"])
plt.axhline(0, linewidth=1)
plt.xticks(
    range(len(df_delta)),
    [f"{m}|{p}" for m, p in zip(df_delta[col_model].astype(str), df_delta[col_prompt].astype(str))],
    rotation=60,
    ha="right",
    fontsize=7,
)
plt.ylabel("Δ FALSE ACTION RATE (TRUE-NOTA) vs baseline")
plt.title("Change in unsafe forced-choice vs baseline (lower is better)")
plt.tight_layout()
plt.savefig(OUTDIR / "fig6_delta_far_vs_baseline.pdf")
plt.close()


# -------------------------
# Load raw per-question outputs (results_raw/*.csv)
# -------------------------
raw_files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
raw_all = []

if raw_files:
    for f in raw_files:
        rdf = pd.read_csv(f)
        fname_model, fname_prompt = parse_model_prompt_from_filename(f)
        model_prefix, c_choice, c_abst, c_conf = detect_raw_cols(rdf)

        model = normalize_model_name(model_prefix if model_prefix else fname_model)
        prompt = normalize_prompt_name(fname_prompt)

        if "question_id" not in rdf.columns or c_choice is None:
            continue

        tmp = pd.DataFrame({
            "question_id": rdf["question_id"],
            "model": model,
            "prompt": prompt,
            "choice": rdf[c_choice],
            "abstain_code": rdf[c_abst] if (c_abst and c_abst in rdf.columns) else "",
            "confidence": rdf[c_conf] if (c_conf and c_conf in rdf.columns) else np.nan,
        })
        raw_all.append(tmp)

raw = pd.concat(raw_all, ignore_index=True) if raw_all else None


# -------------------------
# Raw-based figs (trend hunting)
# -------------------------
if raw is not None and len(raw) > 0:
    raw["prompt_l"] = raw["prompt"].astype(str).map(safe_lower)
    raw["choice"] = raw["choice"].astype(str).str.strip()
    raw["abstain_code"] = raw["abstain_code"].astype(str).replace({"nan": ""}).str.strip()
    raw["confidence"] = pd.to_numeric(raw["confidence"], errors="coerce")

    raw["picked_option"] = raw["choice"].isin(["A", "B", "C", "D"])

    # Define FALSE ACTION behaviorally:
    # picked_option when abstain_code says NO_VALID_OPTION
    raw["false_action"] = raw["picked_option"] & (raw["abstain_code"].str.upper() == "NO_VALID_OPTION")

    # -------------------------
    # FIG 7: Confidence distributions by prompt (boxplot per model)
    # -------------------------
    for model, g in raw.groupby("model"):
        g = g.dropna(subset=["confidence"]).copy()
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

    # -------------------------
    # FIG 8: Abstain-code composition (stacked bars among abstentions)
    # -------------------------
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

    # -------------------------
    # FIG 9: Choice-share (A/B/C/D vs abstain) by (model × prompt)
    # -------------------------
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
                 [c for c in pivot2.columns if c not in ["A","B","C","D","ABSTAIN"]]
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

    # -------------------------
    # FIG 10: Prompt stability (A/B/C/D only)
    # -------------------------
    stability_rows = []
    for model, g in raw.groupby("model"):
        wide = g.pivot_table(index="question_id", columns="prompt", values="choice", aggfunc="first")
        norm = wide.applymap(lambda x: x if x in ["A","B","C","D"] else np.nan)

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

    # -------------------------
    # FIG 11: Mean confidence on FALSE ACTIONS (TRUE-NOTA behavior)
    # -------------------------
    fa = raw[raw["false_action"]].dropna(subset=["confidence"]).copy()
    if not fa.empty:
        fa_mean = (
            fa.groupby(["model", "prompt"])["confidence"]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 5))
        for model, g in fa_mean.groupby("model"):
            g = g.sort_values("prompt", key=lambda s: s.map(prompt_sort_key))
            plt.plot(g["prompt"], g["confidence"], marker="o", label=model)

        plt.ylim(0, 1)
        plt.ylabel("Mean confidence on FALSE ACTIONS")
        plt.xlabel("Prompt")
        plt.xticks(rotation=30, ha="right")
        plt.title("Confidence when models take unsafe actions (FALSE ACTIONS)")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(OUTDIR / "fig11_mean_confidence_false_actions.pdf")
        plt.close()
    else:
        print("[INFO] FIG 11 skipped: no false_action rows found in raw (check abstain_code == NO_VALID_OPTION).")


# -------------------------
# FIG 12: Overconfidence signal (wrong − right confidence gap) — metrics-based
# -------------------------
# This requires aggregated columns in metrics/500q_results2.csv.
# If your metrics file doesn't have them, this figure will be skipped.

col_conf_right = pick_col(dfm, ["mean_confidence_correct", "confidence_correct", "mean_conf_right", "conf_right"])
col_conf_wrong = pick_col(dfm, ["mean_confidence_incorrect", "confidence_incorrect", "mean_conf_wrong", "conf_wrong"])

if col_conf_right and col_conf_wrong:
    dfm[col_conf_right] = dfm[col_conf_right].map(to_rate)
    dfm[col_conf_wrong] = dfm[col_conf_wrong].map(to_rate)
    dfm["overconfidence_gap"] = dfm[col_conf_wrong] - dfm[col_conf_right]

    plt.figure(figsize=(10, 5))
    plt.axhline(0, linewidth=1)

    for model, g in dfm.groupby(col_model):
        g = g.copy()
        g = g.set_index(col_prompt).reindex(prompts_sorted).reset_index()
        plt.plot(g[col_prompt], g["overconfidence_gap"], marker="o", label=str(model))

    plt.ylabel("Confidence gap (wrong − right)")
    plt.xlabel("Prompt")
    plt.xticks(rotation=30, ha="right")
    plt.title("Overconfidence signal across prompts (metrics-based)")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUTDIR / "fig12_overconfidence_gap.pdf")
    plt.close()
else:
    print("[INFO] FIG 12 skipped: metrics file lacks mean confidence right/wrong columns.")


# -------------------------
# FIG 13: FALSE ACTION RATE grouped bars (matches your screenshot intent)
# -------------------------
prompt_order = ["answer-then-double-check", "baseline", "safety-prompt", "think-then-decide"]
models = models_sorted
x = np.arange(len(models))
w = 0.18

plt.figure(figsize=(11, 5))

present_prompts = [p for p in prompt_order if p in dfm[col_prompt].unique()]
for i, p in enumerate(present_prompts):
    vals = (
        dfm[dfm[col_prompt] == p]
        .set_index(col_model)
        .reindex(models)[col_far]
        .values
    )
    plt.bar(x + (i - len(present_prompts)/2)*w, vals, w, label=p)

plt.xticks(x, models, rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("False-action rate on TRUE-NOTA")
plt.title("Forced-choice behavior under missing answers (FALSE-ACTION RATE)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig13_false_action_rate_grouped.pdf")
plt.close()


print(f"\n✅ Saved figures to: {OUTDIR}\n")
