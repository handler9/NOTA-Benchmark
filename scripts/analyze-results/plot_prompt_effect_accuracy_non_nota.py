"""
plot_prompt_effect_accuracy_non_nota.py

Line plot (like your abstention plot) but:
- Y = INTACT / non-NOTA accuracy (%)
- X = prompt strategy (baseline, safety-prompt, answer-then-double-check, think-then-decide)
- One line per model

PLUS: Confidence-by-prompt figure
- Figure A: Mean confidence on correct answers by prompt (bar chart)
- Figure B: Mean confidence on false actions by prompt (bar chart)

Input:
- metrics/500q_results2.csv

Output:
- figures/fig_prompt_effect_accuracy_non_nota_<stamp>.pdf
- figures/fig_prompt_effect_accuracy_non_nota_<stamp>.png
- figures/fig_confidence_by_prompt_correct_<stamp>.pdf
- figures/fig_confidence_by_prompt_correct_<stamp>.png
- figures/fig_confidence_by_prompt_false_actions_<stamp>.pdf
- figures/fig_confidence_by_prompt_false_actions_<stamp>.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Repo-root detection
# -------------------------
ROOT = Path(__file__).resolve()
while ROOT != ROOT.parent and not (ROOT / "metrics").exists():
    ROOT = ROOT.parent
if not (ROOT / "metrics").exists():
    ROOT = Path(__file__).resolve().parents[2]

METRICS = ROOT / "metrics" / "500q_results2.csv"
OUTDIR = ROOT / "figures"
OUTDIR.mkdir(exist_ok=True, parents=True)

STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


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
    if pl == "baseline":
        return "baseline"
    if pl in ["safety", "safetyprompt"]:
        return "safety-prompt"
    if pl in ["doublecheck", "answerthendoublecheck", "answerthendoublecheckprompt"]:
        return "answer-then-double-check"
    if pl in ["think", "thinkthendecide", "thinkthendecideprompt"]:
        return "think-then-decide"
    return str(p)

def to_rate(x):
    """Return float in [0,1] whether input is '82%', 0.82, or 82."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    v = float(s)
    if v > 1.0:
        return v / 100.0
    return v


# -------------------------
# Load + normalize data
# -------------------------
df = pd.read_csv(METRICS)

col_model  = pick_col(df, ["model", "model_name"])
col_prompt = pick_col(df, ["prompt", "prompt_name", "setting"])

# "non-NOTA questions" == "INTACT" in your file naming
col_acc    = pick_col(df, ["intact_accuracy", "accuracy_intact", "acc_intact"])

# confidence columns (present in your 500q_results2.csv)
col_conf_correct = pick_col(df, ["mean_conf_correct", "conf_mean_correct", "mean_confidence_correct"])
col_conf_false_actions = pick_col(df, ["mean_conf_false_actions", "conf_mean_false_actions", "mean_confidence_false_actions"])

if not all([col_model, col_prompt, col_acc]):
    raise ValueError(
        f"Missing required columns. Found: model={col_model}, prompt={col_prompt}, acc={col_acc}. "
        f"Columns in file: {list(df.columns)}"
    )

df[col_model] = df[col_model].map(normalize_model_name)
df[col_prompt] = df[col_prompt].map(normalize_prompt_name)
df[col_acc] = df[col_acc].map(to_rate)

# Enforce prompt order
prompts = ["baseline", "safety-prompt", "answer-then-double-check", "think-then-decide"]
df = df[df[col_prompt].isin(prompts)].copy()
df[col_prompt] = pd.Categorical(df[col_prompt], categories=prompts, ordered=True)


# =========================
# Figure 1: Accuracy lines
# =========================
acc_df = df[[col_model, col_prompt, col_acc]].dropna()

# If there are duplicates, average them (safe default)
plot_df = (
    acc_df.groupby([col_model, col_prompt], as_index=False)[col_acc]
          .mean()
)

# Pivot to wide for line plot
wide = plot_df.pivot(index=col_prompt, columns=col_model, values=col_acc).sort_index()

fig, ax = plt.subplots(figsize=(10, 5.5))

for model in wide.columns:
    ax.plot(
        prompts,
        (wide[model] * 100).values,
        marker="o",
        linewidth=2,
        label=model,
    )

ax.set_ylim(0, 100)
ax.set_ylabel("Accuracy on non-NOTA / INTACT questions (%)")
ax.set_xlabel("Prompt")
ax.set_title("Prompt effect on accuracy (non-NOTA / INTACT)")

ax.set_xticks(range(len(prompts)))
ax.set_xticklabels(prompts, rotation=25, ha="right")

ax.grid(True, axis="y", alpha=0.3)
ax.legend(loc="best")

fig.tight_layout()

out_pdf = OUTDIR / f"fig_prompt_effect_accuracy_non_nota_{STAMP}.pdf"
out_png = OUTDIR / f"fig_prompt_effect_accuracy_non_nota_{STAMP}.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=200)
plt.close(fig)

print(f"\n✅ Saved:\n- {out_pdf}\n- {out_png}\n")


# ==========================================
# Figure 2: Mean confidence by prompt (bars)
# ==========================================
def plot_confidence_by_prompt(
    df_in: pd.DataFrame,
    conf_col: str,
    title: str,
    ylabel: str,
    out_prefix: str,
):
    if conf_col is None or conf_col not in df_in.columns:
        print(f"⚠️ Skipping {out_prefix}: confidence column not found.")
        return

    tmp = df_in[[col_prompt, conf_col]].dropna().copy()

    # Average across models by default (one bar per prompt)
    agg = (
        tmp.groupby(col_prompt, as_index=False)[conf_col]
           .mean()
           .sort_values(col_prompt)
    )

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(agg[col_prompt].astype(str), agg[conf_col].astype(float))

    ax.set_title(title)
    ax.set_xlabel("Prompt")
    ax.set_ylabel(ylabel)

    ax.set_ylim(0, 1.0)
    ax.set_xticks(range(len(prompts)))
    ax.set_xticklabels(prompts, rotation=25, ha="right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()

    out_pdf = OUTDIR / f"{out_prefix}_{STAMP}.pdf"
    out_png = OUTDIR / f"{out_prefix}_{STAMP}.png"
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    print(f"✅ Saved:\n- {out_pdf}\n- {out_png}\n")


plot_confidence_by_prompt(
    df_in=df,
    conf_col=col_conf_correct,
    title="Mean self-reported confidence (correct answers) by prompt",
    ylabel="Mean confidence (0–1)",
    out_prefix="fig_confidence_by_prompt_correct",
)

plot_confidence_by_prompt(
    df_in=df,
    conf_col=col_conf_false_actions,
    title="Mean self-reported confidence (false actions on TRUE-NOTA) by prompt",
    ylabel="Mean confidence (0–1)",
    out_prefix="fig_confidence_by_prompt_false_actions",
)
