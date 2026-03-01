#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# File location (your case = metrics folder)
# --------------------------------------------------
INPUT_PATH = Path("metrics/question_key_TRUE_NOTA_audit2.csv")
OUTPUT_DIR = Path("metrics")

print(f"Loading: {INPUT_PATH}")

# --------------------------------------------------
# Robust load (fixes UnicodeDecodeError)
# --------------------------------------------------
try:
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(INPUT_PATH, encoding="latin1")  # Excel-safe fallback

print("\nBefore:")
print(df["judge_label"].value_counts(dropna=False))

# --------------------------------------------------
# Remove FALSE_NOTA (order preserved)
# --------------------------------------------------
df_clean = df[df["judge_label"] != "FALSE_NOTA"].copy()

print("\nAfter:")
print(df_clean["judge_label"].value_counts(dropna=False))

# --------------------------------------------------
# Save NEW file (no overwrite)
# --------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = OUTPUT_DIR / f"question_key_TRUE_NOTA_clean_{timestamp}.csv"

df_clean.to_csv(output_path, index=False, encoding="utf-8")

print(f"\n✅ Saved to: {output_path}")
print(f"Rows removed: {len(df) - len(df_clean)}")
