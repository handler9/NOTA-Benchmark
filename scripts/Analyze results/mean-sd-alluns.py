import pandas as pd

INPUT_CSV = "all_runs_metrics.csv"
OUTPUT_CSV = "accuracy_overall_mean_sd_table.csv"

df = pd.read_csv(INPUT_CSV)

# Required columns in YOUR file
required = {"model", "prompt", "accuracy_overall"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

# Ensure numeric
df["accuracy_overall"] = pd.to_numeric(df["accuracy_overall"], errors="coerce")

# Aggregate across runs (run-to-run mean ± SD)
summary = (
    df.groupby(["model", "prompt"])["accuracy_overall"]
      .agg(
          **{
              "Accuracy Overall Mean": "mean",
              "Accuracy Std Dev": "std",
          }
      )
      .reset_index()
      .rename(columns={"model": "Model", "prompt": "Prompt"})
)

# Formatting like your screenshot
summary["Accuracy Overall Mean"] = summary["Accuracy Overall Mean"].round(4)
summary["Accuracy Std Dev"] = summary["Accuracy Std Dev"].round(4)

# Optional: nicer ordering
summary = summary.sort_values(["Model", "Prompt"]).reset_index(drop=True)

summary.to_csv(OUTPUT_CSV, index=False)

print(summary)
print(f"\nSaved to: {OUTPUT_CSV}")
