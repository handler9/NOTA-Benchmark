"""
Convert healthbench_main.jsonl to a flat CSV for browsing.
One row per example. Rubrics are summarized as counts and joined text.
"""

import json
import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "healthbench"
IN_FILE  = DATA_DIR / "healthbench_main.jsonl"
OUT_FILE = DATA_DIR / "healthbench_main.csv"


def flatten(row):
    # Last prompt turn (the patient question)
    prompt_turns = row.get("prompt", [])
    question = prompt_turns[-1]["content"] if prompt_turns else ""

    tags = row.get("example_tags", [])
    # Split physician_agreed_category tags from theme tags
    themes = [t.replace("theme:", "") for t in tags if t.startswith("theme:")]
    phys_cats = [t.replace("physician_agreed_category:", "") for t in tags
                 if t.startswith("physician_agreed_category:")]

    rubrics = row.get("rubrics", [])
    n_rubrics = len(rubrics)
    pos_rubrics = [r for r in rubrics if r.get("points", 0) > 0]
    neg_rubrics = [r for r in rubrics if r.get("points", 0) < 0]

    # Summarize top positive and negative criteria
    pos_rubrics_sorted = sorted(pos_rubrics, key=lambda r: -r["points"])
    neg_rubrics_sorted = sorted(neg_rubrics, key=lambda r: r["points"])

    top_pos = " | ".join(r["criterion"][:120] for r in pos_rubrics_sorted[:3])
    top_neg = " | ".join(r["criterion"][:120] for r in neg_rubrics_sorted[:3])

    rubric_axes = set()
    for r in rubrics:
        for t in r.get("tags", []):
            if t.startswith("axis:"):
                rubric_axes.add(t.replace("axis:", ""))

    return {
        "prompt_id": row.get("prompt_id", ""),
        "question": question,
        "themes": ", ".join(themes),
        "physician_agreed_categories": ", ".join(phys_cats),
        "n_rubrics": n_rubrics,
        "n_positive_criteria": len(pos_rubrics),
        "n_negative_criteria": len(neg_rubrics),
        "rubric_axes": ", ".join(sorted(rubric_axes)),
        "top_positive_criteria": top_pos,
        "top_negative_criteria": top_neg,
    }


with open(IN_FILE) as f:
    rows = [json.loads(line) for line in f]

flat = [flatten(r) for r in rows]

with open(OUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=flat[0].keys())
    writer.writeheader()
    writer.writerows(flat)

print(f"Saved {len(flat):,} rows → {OUT_FILE}")
