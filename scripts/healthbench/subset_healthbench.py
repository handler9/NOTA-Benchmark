"""
Subset HealthBench into four experimental conditions.
Reports counts, then samples N_PER_SUBSET per group and saves to data/healthbench/.
"""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "healthbench"
IN_FILE  = DATA_DIR / "healthbench_main.jsonl"

N_PER_SUBSET = 100
SEED = 42

SUBSETS = {
    "underspecified_low_risk": {
        "description": "Underspecified, low-risk — model should seek clarification before answering",
        "require_all": ["theme:context_seeking", "physician_agreed_category:not-enough-context"],
    },
    "underspecified_urgent": {
        "description": "Underspecified, potentially urgent — model should escalate or ask before advising",
        "require_all": ["theme:emergency_referrals", "physician_agreed_category:conditionally-emergent"],
    },
    "sufficient_context": {
        "description": "Sufficient context controls — model has enough info, should answer directly",
        "require_any": ["physician_agreed_category:enough-context", "physician_agreed_category:no-uncertainty"],
        "exclude_any": ["theme:emergency_referrals"],
    },
    "hedging_reducible_uncertainty": {
        "description": "Reducible uncertainty — model should hedge and/or seek clarification, not commit",
        "require_all": ["theme:hedging", "physician_agreed_category:any-reducible-uncertainty"],
    },
}


def has_tag(row, tag):
    return tag in row.get("example_tags", [])


def matches(row, spec):
    tags = row.get("example_tags", [])
    if "require_all" in spec:
        if not all(t in tags for t in spec["require_all"]):
            return False
    if "require_any" in spec:
        if not any(t in tags for t in spec["require_any"]):
            return False
    if "exclude_any" in spec:
        if any(t in tags for t in spec["exclude_any"]):
            return False
    return True


with open(IN_FILE) as f:
    rows = [json.loads(line) for line in f]

random.seed(SEED)

print(f"Total rows: {len(rows):,}\n")
print(f"Target per subset: {N_PER_SUBSET}\n")

all_ok = True
selected_ids = set()

for name, spec in SUBSETS.items():
    pool = [r for r in rows if matches(r, spec) and r["prompt_id"] not in selected_ids]
    n = len(pool)
    status = "OK" if n >= N_PER_SUBSET else f"SHORT — only {n} available"
    print(f"[{name}]")
    print(f"  {spec['description']}")
    print(f"  Matching rows: {n}  →  {status}")

    if n < N_PER_SUBSET:
        all_ok = False
        sample = pool  # take all
    else:
        sample = random.sample(pool, N_PER_SUBSET)

    for r in sample:
        selected_ids.add(r["prompt_id"])

    # Add subset label to each row
    for r in sample:
        r["subset"] = name

    out_path = DATA_DIR / f"healthbench_{name}.jsonl"
    with open(out_path, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(sample)} rows → {out_path.name}\n")

if all_ok:
    print(f"All subsets have {N_PER_SUBSET} rows. Total: {N_PER_SUBSET * len(SUBSETS)}")
else:
    print("Some subsets are short — see above.")

# Also save a combined file with subset label
combined_path = DATA_DIR / "healthbench_subset_combined.jsonl"
with open(IN_FILE) as f:
    all_rows = [json.loads(line) for line in f]

subset_ids = {}
for name, spec in SUBSETS.items():
    out_path = DATA_DIR / f"healthbench_{name}.jsonl"
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            subset_ids[r["prompt_id"]] = name

combined = [r for r in all_rows if r["prompt_id"] in subset_ids]
for r in combined:
    r["subset"] = subset_ids[r["prompt_id"]]

with open(combined_path, "w") as f:
    for r in combined:
        f.write(json.dumps(r) + "\n")

print(f"\nCombined file: {combined_path.name} ({len(combined)} rows)")
