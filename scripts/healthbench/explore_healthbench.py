"""
Explore the HealthBench dataset structure.
Prints schema, tag distributions, category breakdown, and example rows.
Run after download_healthbench.py.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "healthbench"
MAIN_FILE = DATA_DIR / "healthbench_main.jsonl"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


rows = load_jsonl(MAIN_FILE)
print(f"Total rows (main split): {len(rows):,}\n")

# --- Schema ---
first = rows[0]
print("=== Schema (keys in first row) ===")
for k, v in first.items():
    t = type(v).__name__
    if isinstance(v, list) and v:
        t = f"list[{type(v[0]).__name__}] (len={len(v)})"
    elif isinstance(v, dict):
        t = f"dict (keys: {list(v.keys())})"
    print(f"  {k}: {t}")

# --- Categories (if present) ---
if "category" in rows[0]:
    print("\n=== Categories ===")
    cats = Counter(r["category"] for r in rows)
    for cat, n in cats.most_common():
        print(f"  {cat}: {n}")

# --- Example tags distribution (top 30) ---
print("\n=== Top 30 example_tags ===")
tag_counts = Counter()
for r in rows:
    for tag in r.get("example_tags", []):
        tag_counts[tag] += 1
for tag, n in tag_counts.most_common(30):
    print(f"  {tag}: {n}")

# --- Rubric criterion tags (top 30) ---
print("\n=== Top 30 rubric criterion tags ===")
rubric_tag_counts = Counter()
for r in rows:
    for rub in r.get("rubrics", []):
        for tag in rub.get("tags", []):
            rubric_tag_counts[tag] += 1
for tag, n in rubric_tag_counts.most_common(30):
    print(f"  {tag}: {n}")

# --- Points distribution ---
print("\n=== Rubric points distribution ===")
pts = Counter()
for r in rows:
    for rub in r.get("rubrics", []):
        pts[rub.get("points")] += 1
for p, n in sorted(pts.items()):
    print(f"  points={p}: {n} criteria")

# --- Example row ---
print("\n=== Example row ===")
ex = rows[0]
if "category" in ex:
    print(f"Category: {ex['category']}")
print(f"Example tags: {ex['example_tags']}")
print(f"Prompt (last turn): {ex['prompt'][-1]['content'][:300]}...")
print(f"Rubrics ({len(ex['rubrics'])} total):")
for rub in ex["rubrics"][:5]:
    print(f"  [{rub['points']:+d}] {rub['criterion'][:100]}  tags={rub['tags']}")
if len(ex["rubrics"]) > 5:
    print(f"  ... and {len(ex['rubrics']) - 5} more")

# --- Relevant tags for our experiment ---
RELEVANT_TAGS = [
    "clarification", "escalation", "follow-up", "safety", "uncertainty",
    "premature", "defer", "referral", "emergency", "false premise",
    "appropriate", "unnecessary", "over-caution", "reassurance",
]
print("\n=== Tags relevant to premature closure experiment ===")
for tag in RELEVANT_TAGS:
    n_ex = sum(1 for r in rows if tag in r.get("example_tags", []))
    n_rub = rubric_tag_counts.get(tag, 0)
    if n_ex > 0 or n_rub > 0:
        print(f"  '{tag}': {n_ex} examples, {n_rub} rubric criteria")
