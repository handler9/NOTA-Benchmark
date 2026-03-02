"""
make_all_figures_500q.py  (AfriMedQA — abstention version + extra figures)

Generates paper-ready figures from:
- metrics/afrimedqa_results.csv
- results_raw_afrimedqa/*.csv (optional; only used if present)

Outputs: figures/<unique_run_dir>/*.pdf
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Non-interactive backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ------------------------------------------------------
# Repo root detection
# ------------------------------------------------------
ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "metrics").exists():
    ROOT = ROOT.parent

if not (ROOT / "metrics").exists():
    ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------
# Paths
# ------------------------------------------------------
METRICS_CANDIDATES = [
    ROOT / "metrics" / "afrimedqa_results.csv",
    ROOT / "metrics" / "africamedqa_results_1.csv",
]
METRICS = next((p for p in METRICS_CANDIDATES if p.exists()), METRICS_CANDIDATES[0])

RAW_DIR = ROOT / "results_raw_afrimedqa"

RUN_STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = ROOT / "figures" / f"afrimedqa_{RUN_STAMP}"
OUTDIR.mkdir(exist_ok=False, parents=True)


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

def strip_dataset_suffix(stem: str):
    for suf in ["_afrimedqa", "_africamedqa"]:
        if stem.lower().endswith(suf):
            return stem[: -len(suf)]
    return stem

def parse_model_prompt_from_filename(path: Path):
    stem = strip_dataset_suffix(path.stem)
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "unknown"

def detect_raw_cols(df):
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
        return "GPT-5" if "5" in sl else "GPT"
    if "claude" in sl:
        return "Claude"
    if "deepseek" in sl:
        return "DeepSeek"
    if "llama" in sl:
        return "Llama"
    return s

def normalize_prompt_name(p):
    pl = safe_lower(p).replace("-", "").replace("_", "").replace(" ", "")
    if pl in ["baseline"]:
        return "baseline"
    if pl in ["safety", "safetyprompt"]:
        return "safety-prompt"
    if pl in ["doublecheck", "answerthendoublecheck", "answerthendoublecheckprompt", "doublecheckprompt"]:
        return "answer-then-double-check"
    if pl in ["think", "thinkthendecide", "thinkthendecideprompt"]:
        return "think-then-decide"
    return str(p)

def prompt_sort_key(p):
    order = {
        "baseline": 0,
        "safety-prompt": 1,
        "answer-then-double-check": 2,
        "think-then-decide": 3,
    }
    return order.get(normalize_prompt_name(p), 999)

def to_rate(x):
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

def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


# -------------------------
# Load metrics
# -------------------------
if not METRICS.exists():
    raise FileNotFoundError(f"Missing metrics file: {METRICS}")

dfm = pd.read_csv(METRICS)

col_model  = pick_col(dfm, ["model", "model_name"])
col_prompt = pick_col(dfm, ["prompt", "prompt_name", "setting"])
col_acc    = pick_col(dfm, ["intact_accuracy", "accuracy_intact", "acc_intact"])
col_abst_true_nota = pick_col(dfm, ["abstain_rate_true_nota"])

need = [
    ("model", col_model),
    ("prompt", col_prompt),
    ("intact_accuracy", col_acc),
    ("abstain_rate_true_nota", col_abst_true_nota),
]
missing = [n for n, c in need if c is None]
if missing:
    raise ValueError(
        f"Missing columns in {METRICS} for: {missing}\n"
        f"Columns present: {list(dfm.columns)}"
    )

dfm[col_model] = dfm[col_model].map(normalize_model_name)
dfm[col_prompt] = dfm[col_prompt].map(normalize_prompt_name)

dfm[col_acc] = dfm[col_acc].map(to_rate)
dfm[col_abst_true_nota] = dfm[col_abst_true_nota].map(to_rate)

prompts_sorted = sorted(dfm[col_prompt].astype(str).unique(), key=prompt_sort_key)
models_sorted = sorted(dfm[col_model].astype(str).unique())

N_LABEL = "AfriMedQA"


# ======================================================
# FIG 1: Overall grouped bars (baseline prompt)
# ======================================================
PROMPT_TO_PLOT = "baseline"
sub = dfm[dfm[col_prompt].astype(str).eq(PROMPT_TO_PLOT)].copy()
if sub.empty:
    first = dfm[col_prompt].astype(str).iloc[0]
    sub = dfm[dfm[col_prompt].astype(str).eq(first)].copy()
    PROMPT_TO_PLOT = first

sub = sub.sort_values(col_model)
x = range(len(sub))
w = 0.4

plt.figure(figsize=(10, 5))
plt.bar([i - w/2 for i in x], sub[col_acc], w, label="INTACT accuracy")
plt.bar([i + w/2 for i in x], sub[col_abst_true_nota], w, label="TRUE-NOTA abstention rate")
plt.xticks(list(x), sub[col_model], rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("Rate")
plt.title(f"{N_LABEL} overall performance (prompt = {PROMPT_TO_PLOT})")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig1_overall.pdf")
plt.close()


# ======================================================
# FIG 2: Prompt effect (abstention)
# ======================================================
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
plt.title(f"Prompt effect on abstention behavior ({N_LABEL})")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(OUTDIR / "fig2_prompt_effect.pdf")
plt.close()


# ======================================================
# FIG 3: Accuracy–Abstention tradeoff (simple labeled scatter)
# ======================================================
plt.figure(figsize=(7, 6))
for _, r in dfm.iterrows():
    plt.scatter(r[col_acc], r[col_abst_true_nota])
    plt.text(r[col_acc], r[col_abst_true_nota], f"{r[col_model]}|{r[col_prompt]}", fontsize=7)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("INTACT accuracy")
plt.ylabel("TRUE-NOTA abstention rate")
plt.title(f"Accuracy–Abstention tradeoff (model × prompt) — {N_LABEL}")
plt.tight_layout()
plt.savefig(OUTDIR / "fig3_tradeoff.pdf")
plt.close()


# ======================================================
# FIG 4: Heatmap of abstention
# ======================================================
pivot_abst = dfm.pivot_table(index=col_model, columns=col_prompt,
                             values=col_abst_true_nota, aggfunc="mean")
pivot_abst = pivot_abst.reindex(index=models_sorted, columns=prompts_sorted)

plt.figure(figsize=(max(8, len(prompts_sorted)*1.3),
                    max(4, len(models_sorted)*0.7)))
plt.imshow(pivot_abst.values, aspect="auto", vmin=0, vmax=1)
plt.xticks(range(len(pivot_abst.columns)),
           pivot_abst.columns.astype(str),
           rotation=30, ha="right")
plt.yticks(range(len(pivot_abst.index)),
           pivot_abst.index.astype(str))
plt.colorbar(label="TRUE-NOTA abstention rate")
plt.title(f"Heatmap: TRUE-NOTA abstention rate (model × prompt) — {N_LABEL}")
plt.tight_layout()
plt.savefig(OUTDIR / "fig4_heatmap_abstention.pdf")
plt.close()


# ======================================================
# FIG 6: Delta abstention vs baseline
# ======================================================
baseline_map = (
    dfm[dfm[col_prompt].astype(str).eq("baseline")]
    .set_index(col_model)[col_abst_true_nota]
    .to_dict()
)

df_delta = dfm.copy()
df_delta["baseline_abst"] = df_delta[col_model].map(baseline_map)
df_delta["delta_abst_vs_baseline"] = (
    df_delta[col_abst_true_nota] - df_delta["baseline_abst"]
)
df_delta = df_delta[~df_delta["baseline_abst"].isna()].copy()

plt.figure(figsize=(10, 6))
plt.scatter(range(len(df_delta)), df_delta["delta_abst_vs_baseline"])
plt.axhline(0, linewidth=1)
plt.xticks(
    range(len(df_delta)),
    [f"{m}|{p}" for m, p in zip(df_delta[col_model], df_delta[col_prompt])],
    rotation=60, ha="right", fontsize=7,
)
plt.ylabel("Δ abstention rate vs baseline")
plt.title(f"Change in abstention vs baseline — {N_LABEL}")
plt.tight_layout()
plt.savefig(OUTDIR / "fig6_delta_abstention_vs_baseline.pdf")
plt.close()


# ======================================================
# FIG 13: Grouped abstention bars
# ======================================================
prompt_order = [
    "answer-then-double-check",
    "baseline",
    "safety-prompt",
    "think-then-decide",
]

models = models_sorted
x = np.arange(len(models))
w = 0.18

plt.figure(figsize=(11, 5))

present_prompts = [p for p in prompt_order if p in dfm[col_prompt].unique()]
for i, p in enumerate(present_prompts):
    vals = (
        dfm[dfm[col_prompt] == p]
        .set_index(col_model)
        .reindex(models)[col_abst_true_nota]
        .values
    )
    plt.bar(x + (i - len(present_prompts)/2)*w, vals, w, label=p)

plt.xticks(x, models, rotation=30, ha="right")
plt.ylim(0, 1)
plt.ylabel("Abstention rate on TRUE-NOTA")
plt.title(f"Abstention behavior under missing answers — {N_LABEL}")
plt.legend()
plt.tight_layout()
plt.savefig(OUTDIR / "fig13_abstention_rate_grouped.pdf")
plt.close()


# ======================================================
# FIG 14 (NEW): Accuracy vs Abstention across prompt strategies
#   - color = prompt
#   - marker = model
#   - Pearson r shown
# ======================================================
# Percent scale for plotting
acc_pct = dfm[col_acc] * 100.0
abst_pct = dfm[col_abst_true_nota] * 100.0
r = pearson_r(acc_pct, abst_pct)

prompt_levels = prompts_sorted
model_levels = models_sorted

# marker map by model (matches your example vibe)
marker_map = {
    "Claude": "o",
    "DeepSeek": "s",
    "GPT-5": "^",
    "GPT": "^",
    "Gemini": "D",
    "Llama": "P",  # filled plus-ish
}
default_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
for i, m in enumerate(model_levels):
    if m not in marker_map:
        marker_map[m] = default_markers[i % len(default_markers)]

# color map by prompt using matplotlib default cycle
cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
if not cycle:
    cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]
color_map = {p: cycle[i % len(cycle)] for i, p in enumerate(prompt_levels)}

plt.figure(figsize=(10, 7))

# plot points
for _, row in dfm.iterrows():
    p = str(row[col_prompt])
    m = str(row[col_model])
    plt.scatter(
        row[col_acc] * 100.0,
        row[col_abst_true_nota] * 100.0,
        marker=marker_map.get(m, "o"),
        s=90,
        c=color_map.get(p, "C0"),
        edgecolors="black",
        linewidths=0.4,
        alpha=0.9,
    )

# prompt legend (colors)
prompt_handles = [
    Line2D([0], [0], marker="o", linestyle="",
           markerfacecolor=color_map[p], markeredgecolor="black",
           markersize=9, label=p)
    for p in prompt_levels
]
leg1 = plt.legend(handles=prompt_handles, title="Prompt", loc="upper left", frameon=True)
plt.gca().add_artist(leg1)

# model legend (markers)
model_handles = [
    Line2D([0], [0], marker=marker_map[m], linestyle="",
           markerfacecolor="black", markeredgecolor="black",
           markersize=9, label=m)
    for m in model_levels
]
plt.legend(handles=model_handles, title="Model", loc="lower right", frameon=True)

plt.xlim(70, 100)
plt.ylim(0, 100)
plt.xlabel("INTACT accuracy (%)")
plt.ylabel("Abstention rate on TRUE-NOTA (%)")
plt.title(f"Accuracy vs Abstention Across Prompt Strategies\nPearson r = {r:.2f}")
plt.tight_layout()
plt.savefig(OUTDIR / "fig14_accuracy_vs_abstention_by_prompt_and_model.pdf")
plt.close()


# ======================================================
# FIG 15 (NEW): Prompt effect on accuracy (INTACT)
#   - one line per model across prompts
# ======================================================
plt.figure(figsize=(10, 5))
for model, g in dfm.groupby(col_model):
    g = g.copy()
    g = g.set_index(col_prompt).reindex(prompts_sorted).reset_index()
    plt.plot(
        g[col_prompt].astype(str),
        g[col_acc] * 100.0,
        marker="o",
        label=str(model),
    )

plt.ylim(0, 100)
plt.ylabel("Accuracy on INTACT questions (%)")
plt.xlabel("Prompt")
plt.xticks(rotation=30, ha="right")
plt.title("Prompt effect on accuracy (non-NOTA / INTACT)")
plt.legend(loc="lower left", frameon=True)
plt.tight_layout()
plt.savefig(OUTDIR / "fig15_prompt_effect_on_accuracy_intact.pdf")
plt.close()


print(f"\n✅ Saved figures to: {OUTDIR}\n")
print(f"[INFO] Metrics used: {METRICS}")
print(f"[INFO] Raw dir used: {RAW_DIR}")
