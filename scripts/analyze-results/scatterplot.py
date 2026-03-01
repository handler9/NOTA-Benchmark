"""
make_corr_accuracy_abstention.py

Correlation analysis to test whether prompt strategies that preserve INTACT accuracy
also produce high abstention on TRUE-NOTA.

Inputs:
- metrics/500q_results2.csv

Outputs:
- figures/fig_corr_accuracy_vs_abstention.pdf
- figures/fig_corr_prompt_level_accuracy_vs_abstention.pdf
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
    ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "metrics" / "500q_results2.csv"
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
    if pl in ["doublecheck", "answerthendoublecheck", "answerthendoublecheckprompt"]:
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

def pearson_r(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 2:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])

def spearman_r(x, y):
    # Spearman = Pearson correlation of ranks
    x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    return pearson_r(x, y)

def maybe_pvalues(x, y):
    """
    Returns (pearson_p, spearman_p) if scipy is available; else (None, None).
    """
    try:
        from scipy.stats import pearsonr, spearmanr
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2:
            return None, None
        pr = pearsonr(x[m], y[m])
        sr = spearmanr(x[m], y[m])
        return float(pr.pvalue), float(sr.pvalue)
    except Exception:
        return None, None


# -------------------------
# Load metrics
# -------------------------
if not METRICS.exists():
    raise FileNotFoundError(f"Missing metrics file: {METRICS}")

dfm = pd.read_csv(METRICS)

col_model = pick_col(dfm, ["model", "model_name"])
col_prompt = pick_col(dfm, ["prompt", "prompt_name", "setting"])
col_acc = pick_col(dfm, ["intact_accuracy", "accuracy_intact", "acc_intact"])
col_abst = pick_col(dfm, ["abstain_rate_true_nota"])

need = [
    ("model", col_model),
    ("prompt", col_prompt),
    ("intact_accuracy", col_acc),
    ("abstain_rate_true_nota", col_abst),
]
missing = [n for n, c in need if c is None]
if missing:
    raise ValueError(
        f"Missing columns in {METRICS} for: {missing}\n"
        f"Columns present: {list(dfm.columns)}"
    )

# Normalize labels and coerce rates
dfm[col_model] = dfm[col_model].map(normalize_model_name)
dfm[col_prompt] = dfm[col_prompt].map(normalize_prompt_name)
dfm[col_acc] = dfm[col_acc].map(to_rate)
dfm[col_abst] = dfm[col_abst].map(to_rate)

prompts_sorted = sorted(dfm[col_prompt].astype(str).unique(), key=prompt_sort_key)


# -------------------------
# FIG 1: Scatter of model×prompt points + correlations
# X-axis in % and truncated to 70–100%
# -------------------------
corr_df = dfm[[col_model, col_prompt, col_acc, col_abst]].dropna().copy()

r_p = pearson_r(corr_df[col_acc], corr_df[col_abst])
r_s = spearman_r(corr_df[col_acc], corr_df[col_abst])
p_p, p_s = maybe_pvalues(corr_df[col_acc], corr_df[col_abst])

plt.figure(figsize=(7, 6))
for _, r in corr_df.iterrows():
    x = float(r[col_acc]) * 100.0
    y = float(r[col_abst]) * 100.0
    plt.scatter(x, y, alpha=0.85)
    plt.text(x, y, f"{r[col_model]}|{r[col_prompt]}", fontsize=7)

title = "Accuracy vs abstention across prompt strategies"
subtitle = f"Pearson r = {r_p:.2f}, Spearman ρ = {r_s:.2f}"
if p_p is not None and p_s is not None:
    subtitle += f" (p={p_p:.3g}, p={p_s:.3g})"

plt.title(title + "\n" + subtitle)
plt.xlabel("INTACT accuracy (%)")
plt.ylabel("Abstention rate on TRUE-NOTA (%)")
plt.xlim(70, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTDIR / "fig_corr_accuracy_vs_abstention.pdf")
plt.close()


# -------------------------
# FIG 2: Prompt-level summary (mean across models)
# X-axis in % and truncated to 70–100%
# -------------------------
prompt_means = (
    dfm.groupby(col_prompt, as_index=False)
       .agg({col_acc: "mean", col_abst: "mean"})
       .copy()
)

# Keep consistent order
prompt_means["_k"] = prompt_means[col_prompt].map(prompt_sort_key)
prompt_means = prompt_means.sort_values("_k").drop(columns=["_k"])

r_p2 = pearson_r(prompt_means[col_acc], prompt_means[col_abst])
r_s2 = spearman_r(prompt_means[col_acc], prompt_means[col_abst])
p_p2, p_s2 = maybe_pvalues(prompt_means[col_acc], prompt_means[col_abst])

plt.figure(figsize=(6, 5))
plt.scatter(
    prompt_means[col_acc] * 100.0,
    prompt_means[col_abst] * 100.0,
    s=140
)

for _, r in prompt_means.iterrows():
    x = float(r[col_acc]) * 100.0
    y = float(r[col_abst]) * 100.0
    plt.text(x, y, str(r[col_prompt]), fontsize=9, ha="center")

title2 = "Prompt-level accuracy vs abstention (mean across models)"
subtitle2 = f"Pearson r = {r_p2:.2f}, Spearman ρ = {r_s2:.2f}"
if p_p2 is not None and p_s2 is not None:
    subtitle2 += f" (p={p_p2:.3g}, p={p_s2:.3g})"

plt.title(title2 + "\n" + subtitle2)
plt.xlabel("Mean INTACT accuracy (%)")
plt.ylabel("Mean abstention rate on TRUE-NOTA (%)")
plt.xlim(70, 100)
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(OUTDIR / "fig_corr_prompt_level_accuracy_vs_abstention.pdf")
plt.close()


print("\n✅ Saved correlation figures to:")
print(f"  - {OUTDIR / 'fig_corr_accuracy_vs_abstention.pdf'}")
print(f"  - {OUTDIR / 'fig_corr_prompt_level_accuracy_vs_abstention.pdf'}\n")
