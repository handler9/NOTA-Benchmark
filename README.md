# LLM Clinical Safety NOTA Benchmark

This repository contains an evaluation framework for measuring **clinical safety**, **abstention behavior**, and **false-action risk** in large language models (LLMs). The benchmark uses a mixed dataset of INTACT and TRUE-NOTA medical questions to evaluate safe decision-making and refusal behavior.

---

## Benchmark Overview

The benchmark includes:

- **INTACT questions** — one correct answer is present
- **TRUE-NOTA questions** — the correct answer has been removed; the safest action is to abstain

All questions are **shuffled together**, and INTACT questions have one distractor removed so all items contain **exactly four answer choices**.

### Key Metrics

- **Intact Accuracy (%)** — correct answers on INTACT questions
- **NOTA Abstention (%)** — abstention rate on TRUE-NOTA questions
- **Intact Abstention (%)** — abstention rate on INTACT questions

### Prompting Conditions

Each model is evaluated under four prompt styles:

1. **Baseline** — no abstention rules
2. **Safety** — explicit rules to abstain when unsure or unsafe
3. **Think-Then-Decide** — internal reasoning before answering
4. **Answer-Then-Double-Check** — model answers first, then performs a safety review

---

## Models Evaluated

- Gemini 2.5 Pro
- GPT-5
- DeepSeek R1/V1
- Claude 4
- Llama 4 Maverick

---

## Experiments

### 1. Main 500-Question Experiment

500 questions pulled from MedQA (250 INTACT + 250 TRUE-NOTA), shuffled together.

| File/Folder | Path |
|---|---|
| Questions | `data/questions.csv` |
| Answer key | `data/question_key.csv` |
| Model scripts | `scripts/<model-name>/` |
| Raw results | `results_raw/` |
| Analysis script | `scripts/analyze-results/All_Model_Results.py` |
| Metrics output | `metrics/medqa_results.csv` |

---

### 2. Variability Experiment (5 Runs × 500 Questions)

Same 500 questions run 5 times each to test variability across runs (100 total CSVs).

> **Note:** The `llm_runs/` folder is not included in this repository due to size (~131MB). Contact the author for access.

| File/Folder | Path |
|---|---|
| Run CSVs | `llm_runs/` *(not included — see note above)* |
| Results script | `scripts/analyze-results/results_allruns.py` |
| Metrics output | `metrics/all_runs_metrics.csv` |
| Variance script | `scripts/analyze-results/variancebymodel.py` |
| Variance output | `metrics/variance_summary_by_model_prompt.csv` |
| Mean/SD script | `scripts/analyze-results/mean-sd-allruns.py` |
| Mean/SD output | `metrics/accuracy_overall_mean_sd_table.csv` |
| Run manifest | `data/manifest_files_loaded.csv` |

---

### 3. 50 Intact Question Sanity Check

50 INTACT questions pulled from MedQA as a sanity check.

| File/Folder | Path |
|---|---|
| Questions | `data/50questions.csv` |
| Answer key | `data/50question_key.csv` |
| Model scripts | `scripts/50-question-test/` |
| Raw results | `results_raw_50q_test/` |
| Analysis script | `scripts/analyze-results/50questions-analysis.py` |
| Metrics output | `metrics/summary_50intact.csv` |

---

### 4. Clinical Judge Test

Tested GPT-5's ability to classify whether a question was truly TRUE-NOTA, judging against a set of 100 clinician-validated questions.

All files are located in the `clinical-judge-test/` folder.

| File/Folder | Path |
|---|---|
| Questions | `clinical-judge-test/Clinically-annotated-100qs.csv` |
| Judge script (v1) | `clinical-judge-test/clinical-judge-test.py` |
| Raw results (v1) | `clinical-judge-test/Clinically-annotated-100qs_gpt5_true_nota_judged-1.csv` |
| Metrics (v1) | `clinical-judge-test/Clinically-annotated-100qs_gpt5_true_nota_metrics-1.csv` |
| Raw results (v2, different prompt) | `clinical-judge-test/Clinically-annotated-100qs_gpt5_true_nota_judged.csv` |
| F1/Precision/Recall (v2) | `clinical-judge-test/clinical-annotation-judge-test-results.csv` |

---

### 5. AfriMedQA Experiment

490 MCQs from the AfriMedQA dataset, run using the same process as the main experiment.

| File/Folder | Path |
|---|---|
| Questions | `data/afrimedqa_questions.csv` |
| Answer key | `data/afrimedqa_questions_key.csv` |
| Model scripts | `scripts/<model-name>/` *(scripts labeled `afrimedqa`)* |
| Raw results | `results_raw_afrimedqa/` |
| Analysis script | `scripts/analyze-results/Afrimedqa_all_model_results.py` |
| Metrics output | `metrics/afrimedqa_results.csv` |

---

### 6. False-NOTA Removal and Re-runs

A GPT-5 judge was used to identify questions incorrectly classified as TRUE-NOTA ("false NOTAs"). These were manually reviewed, removed, and the experiments re-run on cleaned datasets.

#### MedQA

| File/Folder | Path |
|---|---|
| Judge script | `scripts/Medqa-finding-false-NOTAs.py` |
| Audit output | `metrics/MedQA500_TRUE_NOTA_audit.csv` |
| Cleaned questions (N=466) | `data/questions_positive_nota_only.csv` |
| Cleaned answer key | `data/question_key_positive_nota_only.csv` |
| Raw results | `results_raw_nota_positives/` |
| Analysis script | `scripts/analyze-results/all_model_results-NOTA-POSITIVES.py` |
| Metrics output | `metrics/medqa_nota_positive_results.csv` |

#### AfriMedQA

| File/Folder | Path |
|---|---|
| Judge script | `scripts/Afrimedqa-finding-false-NOTAs.py` |
| Audit output | `metrics/afrimedqa_questions_TRUE_NOTA_audit.csv` |
| Audit (false NOTAs removed) | `metrics/afrimedqa_questions_key_TRUE_NOTA_audit_FALSE-NOTA-REMOVED.csv` |
| Cleaned questions (N=425) | `data/afrimedqa_questions_FALSE-NOTA-REMOVED.csv` |
| Cleaned answer key | `data/afrimedqa_questions_KEY_FALSE-NOTA-REMOVED.csv` |
| Raw results | `results_raw_afrimedqa_nota_positives/` |
| Analysis script | `scripts/analyze-results/afrimedqa-true-positive-results-rounded.py` |
| Metrics output | `metrics/afrimedqa_results_nota-positives.csv` |

---

### 7. Ranked Performance Tables

Generates ranked performance tables for each major dataset using an **abstention-prioritized lexicographic ranking rule**.

#### Ranking Framework

Models are ranked using the following priority order:

1. Abstention rate on TRUE-NOTA questions *(higher is better)*
2. Intact accuracy *(higher is better)*
3. Intact abstention rate *(lower is better)*

For each model, the **best-performing prompt strategy** is selected using the same abstention-first rule before ranking.

| File/Folder | Path |
|---|---|
| Script | `scripts/analyze-results/make_ranked_tables_all_datasets.py` |
| Input | `metrics/medqa_results.csv`, `metrics/afrimedqa_results.csv`, `metrics/afrimedqa_results_nota-positives.csv`, `metrics/medqa_nota_positive_results.csv` |
| CSV tables | `tables/` |
| PNG tables | `tables/` |

**To run:**

```bash
python scripts/analyze-results/make_ranked_tables_all_datasets.py

# CSV output only (no PNG):
python scripts/analyze-results/make_ranked_tables_all_datasets.py --no_png
```

---

## Installation & Setup

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

## Repository Structure

```
NOTA-Benchmark/
├── data/
│   ├── questions.csv
│   ├── question_key.csv
│   ├── 50questions.csv
│   ├── 50question_key.csv
│   ├── afrimedqa_questions.csv
│   ├── afrimedqa_questions_key.csv
│   ├── afrimedqa_questions_FALSE-NOTA-REMOVED.csv
│   ├── afrimedqa_questions_KEY_FALSE-NOTA-REMOVED.csv
│   ├── questions_positive_nota_only.csv
│   └── question_key_positive_nota_only.csv
│
├── scripts/
│   ├── claude-tests/
│   ├── gpt-tests/
│   ├── deepseek-tests/
│   ├── deepseek-reruns/
│   ├── gemini-scripts/
│   ├── llama-tests/
│   ├── 50-question-test/
│   ├── analyze-results/
│   ├── Medqa-finding-false-NOTAs.py
│   ├── Afrimedqa-finding-false-NOTAs.py
│   └── true_nota_judge.py
│
├── results_raw/
├── results_raw_50q_test/
├── results_raw_afrimedqa/
├── results_raw_afrimedqa_nota_positives/
├── results_raw_nota_positives/
│
├── clinical-judge-test/
│
├── metrics/
├── tables/
│
├── environment.yml
├── LICENSE
└── README.md
```

> **Note:** `llm_runs/` is excluded from this repository. Contact the author for access to the full variability run data.

---

## Author

Rebecca Handler
[handler9@stanford.edu](mailto:handler9@stanford.edu)
