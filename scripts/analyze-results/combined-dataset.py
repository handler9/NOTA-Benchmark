import os
import pandas as pd

# --------------------------------------------------------
# ONLY SAFETY-PROMPT across MedQA + AfriMedQA
# with rounding to 3 decimal places
# --------------------------------------------------------

METRICS_DIR = "metrics"

MEDQA_FILE = os.path.join(METRICS_DIR, "500q_results_4.csv")
AFRIMEDQA_FILE = os.path.join(METRICS_DIR, "afrimedqa_results_1.csv")

OUTPUT_FILE = os.path.join(METRICS_DIR, "combined_safety_only1.csv")


def normalize_prompt(p: str) -> str:
    return str(p).strip().lower()


def is_safety_prompt(p: str) -> bool:
    return "safety" in normalize_prompt(p)


def load_and_filter(path: str, dataset_name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")

    df = pd.read_csv(path)

    required_cols = {"model", "prompt"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["dataset"] = dataset_name

    # 🔥 KEEP ONLY SAFETY
    df = df[df["prompt"].apply(is_safety_prompt)].copy()

    # Standardize label
    df["prompt"] = "safety-prompt"

    return df


def round_numeric_columns(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=["float", "float64"]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def main():
    med = load_and_filter(MEDQA_FILE, "MedQA")
    afr = load_and_filter(AFRIMEDQA_FILE, "AfriMedQA")

    combined = pd.concat([med, afr], ignore_index=True)

    # Preferred column order if present
    preferred_cols = [
        "dataset",
        "model",
        "prompt",
        "n_total",
        "n_intact",
        "n_true_nota",
        "accuracy_intact",
        "abstain_rate_true_nota",
        "false_action_rate_true_nota",
        "abstain_rate_intact",
        "accuracy_overall",
    ]
    cols_to_keep = [c for c in preferred_cols if c in combined.columns]
    if cols_to_keep:
        combined = combined[cols_to_keep]

    # ⭐ ROUND TO THIRD DECIMAL
    combined = round_numeric_columns(combined, decimals=3)

    combined = combined.sort_values(["dataset", "model"]).reset_index(drop=True)

    os.makedirs(METRICS_DIR, exist_ok=True)
    combined.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()