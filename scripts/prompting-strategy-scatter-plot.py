"""
make_corr_accuracy_abstention_color_shape.py

Single correlation plot:
- X = INTACT accuracy (%) [70–100]
- Y = Abstention rate on TRUE-NOTA (%)

Encoding (UPDATED):
- Color = prompt strategy
- Marker shape = model

No point labels (legend only).
Timestamped output to avoid overwrite.

Input:
- metrics/500q_results2.csv

Output:
- figures/fig_corr_accuracy_vs_abstention_colorshape_<stamp>.pdf
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
    return str(p).strip()

def to_rate(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s.endswith("%"):
        return float(s[:-1]) / 100
    v = float(s)
    return v / 100 if v > 1 else v

def pearson_r(x, y):
    return float(np.corrcoef(x, y)[0, 1])


# -------------------------
# Load data
# -------------------------
df = pd.read_csv(METRICS)

col_model = pick_col(df, ["model", "model_name"])
col_prompt = pick_col(df, ["prompt", "prompt_name", "setting"])
col_acc = pick_col(df, ["intact_accuracy", "accuracy_intact", "acc_intact"])
col_abst = pick_col(df, ["abstain_rate_true_nota"])

missing = [name for name, col in [
    ("model", col_model),
    ("prompt", col_prompt),
    ("intact_accuracy", col_acc),
    ("abstain_rate_true_nota", col_abst),
] if col is None]
if missing:
    raise ValueError(f"Missing required columns (or aliases) in CSV: {missing}")

df[col_model] = df[col_model].map(normalize_model_name)
df[col_prompt] = df[col_prompt].map(normalize_prompt_name)
df[col_acc] = df[col_acc].map(to_rate)
df[col_abst] = df[col_abst].map(to_rate)

df = df[[col_model, col_prompt, col_acc, col_abst]].dropna()


# -------------------------
# Visual encodings (UPDATED)
# -------------------------
prompts_order = ["baseline", "safety-prompt", "answer-then-double-check", "think-then-decide"]

# Only keep prompts that actually exist in the data, but preserve desired order
prompts_present = [p for p in prompts_order if p in set(df[col_prompt].unique())]
# Include any additional prompts (if any) after the ordered ones
prompts_extra = [p for p in sorted(df[col_prompt].unique()) if p not in set(prompts_present)]
prompts = prompts_present + prompts_extra

models = sorted(df[col_model].unique())

# Color = prompt
colors = dict(zip(prompts, plt.cm.tab10.colors[:len(prompts)]))

# Marker = model
marker_list = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "H"]
markers = dict(zip(models, marker_list[:len(models)]))

# Guardrails: ensure every row can be encoded
unknown_prompts = set(df[col_prompt].unique()) - set(colors.keys())
unknown_models = set(df[col_model].unique()) - set(markers.keys())
if unknown_prompts:
    raise ValueError(f"Found prompts without color mapping: {sorted(unknown_prompts)}")
if unknown_models:
    raise ValueError(f"Found models without marker mapping: {sorted(unknown_models)}")


# -------------------------
# Correlation
# -------------------------
r = pearson_r(df[col_acc], df[col_abst])


# -------------------------
# Plot
# -------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for _, row in df.iterrows():
    ax.scatter(
        row[col_acc] * 100,
        row[col_abst] * 100,
        color=colors[row[col_prompt]],      # UPDATED: color = prompt
        marker=markers[row[col_model]],     # UPDATED: marker = model
        s=90,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.4,
    )

# Axes
ax.set_xlim(70, 100)
ax.set_ylim(0, 100)
ax.set_xticks([70, 75, 80, 85, 90, 95, 100])
ax.set_yticks([0, 20, 40, 60, 80, 100])

ax.set_xlabel("INTACT accuracy (%)")
ax.set_ylabel("Abstention rate on TRUE-NOTA (%)")
ax.set_title(f"Accuracy vs Abstention Across Prompt Strategies\nPearson r = {r:.2f}")

# Prompt legend (colors)
prompt_handles = [
    plt.Line2D([0], [0], marker="o", color="w",
               markerfacecolor=colors[p], label=p, markersize=10)
    for p in prompts
]

# Model legend (shapes)
model_handles = [
    plt.Line2D([0], [0], marker=markers[m], color="black",
               linestyle="", label=m, markersize=9)
    for m in models
]

leg1 = ax.legend(handles=prompt_handles, title="Prompt", loc="upper left")
ax.add_artist(leg1)
ax.legend(handles=model_handles, title="Model", loc="lower right")

fig.tight_layout()

out = OUTDIR / f"fig_corr_accuracy_vs_abstention_colorshape_{STAMP}.pdf"
fig.savefig(out)
plt.close(fig)

print(f"\n✅ Saved: {out}\n")
