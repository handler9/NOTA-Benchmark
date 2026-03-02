# Prompting LLMs to Pause: Guardrails for Enhancing Safety

Evaluation framework for measuring LLM abstention behavior on unanswerable medical questions. Uses a mixed dataset of answerable (INTACT) and unanswerable (TRUE-NOTA) items from MedQA and AfriMed-QA to test whether prompting guardrails can improve safe abstention without reducing accuracy.

---

## Benchmark Overview

- **INTACT questions** — one correct answer is present
- **TRUE-NOTA questions** — the correct answer has been removed; the safest action is to abstain

All questions are shuffled together, and INTACT questions have one distractor removed so all items contain exactly four answer choices.

### Key Metrics

- **Intact Accuracy (%)** — correct answers on INTACT questions
- **NOTA Abstention (%)** — abstention rate on TRUE-NOTA questions
- **Intact Abstention (%)** — abstention rate on INTACT questions

### Prompting Conditions

1. **Baseline** — no abstention rules
2. **Safety** — explicit rules to abstain when unsure or unsafe
3. **Think-Then-Decide** — internal reasoning before answering
4. **Answer-Then-Double-Check** — model answers first, then performs a safety review

### Models Evaluated

- GPT-5
- Gemini 2.5 Pro
- Claude Sonnet 4
- DeepSeek R1/V3
- Llama 4 Maverick

---

## Setup

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate nota
```

### 2. Add API Keys

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GEMINI_API_KEY=your_key
```

---

## Running the Experiments

All scripts should be run from the **repo root**.

### 1. Main MedQA Experiment (500 Questions)

500 questions from MedQA (250 INTACT + 250 TRUE-NOTA), shuffled together.

| File | Path |
|---|---|
| Questions | `data/questions.csv` |
| Answer key | `data/question_key.csv` |
| Model scripts | `scripts/<model-name>/` |
| Raw results | `results_raw/` |
| Analysis script | `scripts/analyze-results/All_Model_Results.py` |
| Metrics output | `metrics/medqa_results.csv` |

**To run:**

```bash
# Run all four prompt variants for each model (example: GPT-5)
python scripts/gpt-tests/gpt_baseline.py
python scripts/gpt-tests/gpt_safety.py
python scripts/gpt-tests/gpt_think.py
python scripts/gpt-tests/gpt_doublecheck.py

# Repeat for other models:
# scripts/claude-tests/  →  claude4_baseline.py, claude4_safety.py, claude4_think.py, claude4_doublecheck.py
# scripts/gemini-scripts/ →  gemini_baseline.py, gemini_safety.py, gemini_think.py, gemini_doublecheck.py
# scripts/deepseek-tests/ →  deepseek_baseline.py, deepseek_safety.py, deepseek_think.py, deepseek_doublecheck.py
# scripts/llama-tests/    →  llama_baseline.py, llama_safety.py, llama_think.py, llama_doublecheck.py

# Analyze results
python scripts/analyze-results/All_Model_Results.py
```

---

### 2. AfriMed-QA Experiment (490 Questions)

490 MCQs from the AfriMedQA dataset, run using the same process as the main experiment.

| File | Path |
|---|---|
| Questions | `data/afrimedqa_questions.csv` |
| Answer key | `data/afrimedqa_questions_key.csv` |
| Model scripts | `scripts/<model-name>/` *(scripts labeled `afrimedqa`)* |
| Raw results | `results_raw_afrimedqa/` |
| Analysis script | `scripts/analyze-results/Afrimedqa_all_model_results.py` |
| Metrics output | `metrics/afrimedqa_results.csv` |

**To run:**

```bash
python scripts/gpt-tests/gpt_afrimedqa.py
python scripts/claude-tests/claude4_Afrimedqa.py
python scripts/gemini-scripts/gemini_afrimedqa.py
python scripts/deepseek-tests/deepseek_afrimedqa.py
python scripts/llama-tests/llama_afrimedqa.py

# Analyze results
python scripts/analyze-results/Afrimedqa_all_model_results.py
```

---

### 3. Ranked Performance Tables

Generates ranked tables for each dataset using an abstention-prioritized ranking rule: 1) NOTA abstention (higher is better), 2) Intact accuracy (higher is better), 3) Intact abstention (lower is better). For each model, the best-performing prompt strategy is selected before ranking.

Output tables (CSV and PNG) are written to `tables/`.

```bash
python scripts/analyze-results/make_ranked_tables_all_datasets.py

# CSV output only (no PNG):
python scripts/analyze-results/make_ranked_tables_all_datasets.py --no_png
```

---

### 4. False-NOTA Removal and Re-runs

A GPT-5 judge identifies questions incorrectly classified as TRUE-NOTA. These are manually reviewed, removed, and experiments re-run on cleaned datasets.

Judge scripts: `scripts/Medqa-finding-false-NOTAs.py`, `scripts/Afrimedqa-finding-false-NOTAs.py`
Cleaned data: `data/questions_positive_nota_only.csv`, `data/afrimedqa_questions_FALSE-NOTA-REMOVED.csv`
Results: `results_raw_nota_positives/`, `results_raw_afrimedqa_nota_positives/`

```bash
# Identify false NOTAs
python scripts/Medqa-finding-false-NOTAs.py
python scripts/Afrimedqa-finding-false-NOTAs.py

# Re-run models on cleaned datasets (scripts labeled positive-nota or false-nota-removed)
# e.g. scripts/gpt-tests/gpt-positive-nota-only.py

# Analyze
python scripts/analyze-results/all_model_results-NOTA-POSITIVES.py
python scripts/analyze-results/afrimedqa-true-positive-results-rounded.py
```

---

### 5. Variability Experiment (5 Runs × 500 Questions)

The same 500 MedQA questions run 5 times per model–prompt condition to assess sampling variability.

> **Note:** The `llm_runs/` folder is not included in this repository (~131MB). Contact the author for access.

```bash
python scripts/analyze-results/results_allruns.py
python scripts/analyze-results/variancebymodel.py
python scripts/analyze-results/mean-sd-allruns.py
```

---

### 6. 50-Question Sanity Check

50 INTACT questions from MedQA used as a sanity check before the main experiment. Scripts are in `scripts/50-question-test/`.

```bash
# Run all models (example: GPT-5)
python scripts/50-question-test/gpt_baseline_50q.py

# Analyze
python scripts/analyze-results/50questions-analysis.py
```

---

### 7. Clinical Judge Test

Tests GPT-5's ability to classify whether a question is truly TRUE-NOTA, validated against 100 clinician-annotated questions. All files are in `clinical-judge-test/`.

```bash
python clinical-judge-test/clinical-judge-test.py
```

---

## Repository Structure

```
NOTA-Benchmark/
├── data/
├── scripts/
│   ├── claude-tests/
│   ├── gpt-tests/
│   ├── gemini-scripts/
│   ├── deepseek-tests/
│   ├── deepseek-reruns/
│   ├── llama-tests/
│   ├── 50-question-test/
│   ├── analyze-results/
│   ├── Medqa-finding-false-NOTAs.py
│   ├── Afrimedqa-finding-false-NOTAs.py
│   └── true_nota_judge.py
├── results_raw/
├── results_raw_50q_test/
├── results_raw_afrimedqa/
├── results_raw_afrimedqa_nota_positives/
├── results_raw_nota_positives/
├── clinical-judge-test/
├── metrics/
├── tables/
├── environment.yml
├── LICENSE
└── README.md
```

> **Note:** `llm_runs/` is excluded from this repository. Contact the author for access to the full variability run data.

---

## Author

Rebecca Handler
[handler9@stanford.edu](mailto:handler9@stanford.edu)
