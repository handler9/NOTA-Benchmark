"""
Download HealthBench from Hugging Face and save locally as JSONL files.
Saves all splits to data/healthbench/.
"""

import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "healthbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "main":      "2025-05-07-06-14-12_oss_eval.jsonl",
    "consensus": "consensus_2025-05-09-20-00-46.jsonl",
    "hard":      "hard_2025-05-08-21-00-10.jsonl",
}

for split_name, source_file in SPLITS.items():
    print(f"Downloading {split_name} ({source_file})...")
    ds = load_dataset(
        "openai/healthbench",
        data_files=source_file,
        split="train",
    )
    out_path = OUT_DIR / f"healthbench_{split_name}.jsonl"
    with open(out_path, "w") as f:
        for row in ds:
            f.write(json.dumps(row) + "\n")
    print(f"  Saved {len(ds):,} rows → {out_path}")

print("\nDone.")
