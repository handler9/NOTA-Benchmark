"""
Compares all model outputs on expanded_afrimedqa against GPT-5 judge labels.
Outputs:
  metrics/expanded_afrimedqa_accuracy.csv   — per-model accuracy/kappa/F1
  metrics/expanded_afrimedqa_confusion.csv  — confusion matrices (long format)
  metrics/expanded_afrimedqa_rowlevel.csv   — row-level labels for all models
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results_raw_1000_afrimedqa_consumer_queries"
JUDGE_PATH  = REPO_ROOT / "metrics" / "gpt_expanded_afrimedqa_judge.csv"
OUT_DIR     = REPO_ROOT / "metrics"

VALID_LABELS = {"ANSWER", "FOLLOW-UP", "DEFER"}

MODELS = {
    "GPT-5 Baseline":    (RESULTS_DIR / "gpt_expanded_afrimedqa_baseline.csv",    "gpt5"),
    "GPT-5 Safety":      (RESULTS_DIR / "gpt_expanded_afrimedqa_safety.csv",       "gpt5"),
    "DeepSeek Baseline": (RESULTS_DIR / "deepseek_expanded_afrimedqa_baseline.csv","deepseek"),
    "DeepSeek Safety":   (RESULTS_DIR / "deepseek_expanded_afrimedqa_safety.csv",  "deepseek"),
    "Gemini Baseline":   (RESULTS_DIR / "gemini_expanded_afrimedqa_baseline.csv",  "gemini"),
    "Gemini Safety":     (RESULTS_DIR / "gemini_expanded_afrimedqa_safety.csv",    "gemini"),
    "Claude Baseline":   (RESULTS_DIR / "claude_expanded_afrimedqa_baseline.csv",  "claude"),
    "Claude Safety":     (RESULTS_DIR / "claude_expanded_afrimedqa_safety.csv",    "claude"),
    "Llama Baseline":    (RESULTS_DIR / "llama_expanded_afrimedqa_baseline.csv",   "llama"),
    "Llama Safety":      (RESULTS_DIR / "llama_expanded_afrimedqa_safety.csv",     "llama"),
}

# ------------------------------------------------------
# Load judge labels
# ------------------------------------------------------
print(f"📂 Loading judge labels from {JUDGE_PATH}")
if not JUDGE_PATH.exists():
    print("❌ Judge file not found. Run gpt_expanded_afrimedqa_judge.py first.")
    raise SystemExit

judge_df = pd.read_csv(JUDGE_PATH)
judge_df = judge_df[judge_df["judge_label"].isin(VALID_LABELS)].copy()
judge_df = judge_df.set_index("row_index")
print(f"   {len(judge_df)} rows with valid judge labels\n")

# ------------------------------------------------------
# Build row-level output
# ------------------------------------------------------
row_df = judge_df[["scenario", "patient_question", "judge_label"]].copy()

accuracy_rows = []
confusion_rows = []

for condition, (path, prefix) in MODELS.items():
    print(f"Processing: {condition}")
    label_col = f"{prefix}_label"

    model_df = pd.read_csv(path)[["row_index", label_col]].copy()
    model_df = model_df.set_index("row_index")

    # Align to rows where judge has valid labels
    merged = judge_df[["judge_label"]].join(model_df, how="inner")
    merged = merged[merged[label_col].isin(VALID_LABELS)]

    y_true = merged["judge_label"].tolist()
    y_pred = merged[label_col].tolist()

    n_total   = len(judge_df)
    n_matched = len(merged)
    n_skipped = n_total - n_matched

    acc   = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    report = classification_report(
        y_true, y_pred,
        labels=["ANSWER", "FOLLOW-UP", "DEFER"],
        output_dict=True,
        zero_division=0,
    )

    accuracy_rows.append({
        "Condition":           condition,
        "N (matched)":         n_matched,
        "N (skipped/errors)":  n_skipped,
        "Accuracy":            round(acc, 4),
        "Cohen Kappa":         round(kappa, 4),
        "F1 ANSWER":           round(report["ANSWER"]["f1-score"], 4),
        "F1 FOLLOW-UP":        round(report["FOLLOW-UP"]["f1-score"], 4),
        "F1 DEFER":            round(report["DEFER"]["f1-score"], 4),
        "Precision ANSWER":    round(report["ANSWER"]["precision"], 4),
        "Precision FOLLOW-UP": round(report["FOLLOW-UP"]["precision"], 4),
        "Precision DEFER":     round(report["DEFER"]["precision"], 4),
        "Recall ANSWER":       round(report["ANSWER"]["recall"], 4),
        "Recall FOLLOW-UP":    round(report["FOLLOW-UP"]["recall"], 4),
        "Recall DEFER":        round(report["DEFER"]["recall"], 4),
        "Macro F1":            round(report["macro avg"]["f1-score"], 4),
        "Weighted F1":         round(report["weighted avg"]["f1-score"], 4),
    })

    # Confusion matrix (long format)
    cm = confusion_matrix(y_true, y_pred, labels=["ANSWER", "FOLLOW-UP", "DEFER"])
    for i, true_label in enumerate(["ANSWER", "FOLLOW-UP", "DEFER"]):
        for j, pred_label in enumerate(["ANSWER", "FOLLOW-UP", "DEFER"]):
            confusion_rows.append({
                "Condition":   condition,
                "True Label":  true_label,
                "Pred Label":  pred_label,
                "Count":       int(cm[i, j]),
            })

    # Add column to row-level df
    col_name = condition.replace(" ", "_").lower() + "_label"
    row_df = row_df.join(
        model_df[label_col].rename(col_name),
        how="left"
    )

# ------------------------------------------------------
# Save outputs
# ------------------------------------------------------
acc_df  = pd.DataFrame(accuracy_rows)
conf_df = pd.DataFrame(confusion_rows)

acc_path  = OUT_DIR / "expanded_afrimedqa_accuracy.csv"
conf_path = OUT_DIR / "expanded_afrimedqa_confusion.csv"
row_path  = OUT_DIR / "expanded_afrimedqa_rowlevel.csv"

acc_df.to_csv(acc_path,   index=False)
conf_df.to_csv(conf_path, index=False)
row_df.to_csv(row_path,   index=True)

print(f"\n✅ Saved:")
print(f"   {acc_path}")
print(f"   {conf_path}")
print(f"   {row_path}")

# ------------------------------------------------------
# Print summary
# ------------------------------------------------------
print("\n" + "="*70)
print("ACCURACY vs. GPT-5 JUDGE")
print("="*70)
print(acc_df[["Condition", "N (matched)", "Accuracy", "Cohen Kappa", "Macro F1"]].to_string(index=False))
