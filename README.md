# Prompting LLMs to Pause: Guardrails for Enhancing Safety in Medical AI

Evaluation framework for studying **premature closure** in large language models — the tendency to commit to a response without sufficient information. Tests whether a safety prompting guardrail can reduce premature closure across two complementary experimental settings: structured MCQs and open-ended clinical conversations.

---

## Study Design

### Experiment 1: MCQ Abstention (MedQA + AfriMedQA)

A mixed dataset of answerable (**INTACT**) and unanswerable (**TRUE-NOTA**) multiple-choice questions tests whether models abstain when the correct answer has been removed.

- **INTACT questions** — one correct answer is present; model should choose it
- **TRUE-NOTA questions** — the correct answer has been removed; the safest action is to abstain

All items have exactly four answer choices (one distractor removed from INTACT items). INTACT and TRUE-NOTA items are shuffled together so models cannot detect the condition.

**Key metrics:**
- **NOTA Abstention (%)** — abstention rate on TRUE-NOTA questions (primary outcome)
- **Intact Accuracy (%)** — correct answers on INTACT questions
- **Intact Abstention (%)** — false abstention rate on INTACT questions (should stay low)

**Datasets:**
- MedQA (500 questions: 250 INTACT + 250 TRUE-NOTA)
- AfriMedQA (490 questions: 245 INTACT + 245 TRUE-NOTA)

---

### Experiment 2: Premature Closure in Open-Ended Conversations (HealthBench)

Multi-turn clinical conversations from HealthBench test whether models commit prematurely to answers when more caution or clarification is warranted.

- **HealthBench** — 861 physician-authored clinical questions across four subcategories:
  - `underspecified_low_risk` (181 questions) — model should seek clarification
  - `underspecified_urgent` (180 questions) — model should escalate or ask before advising
  - `sufficient_context` (250 questions) — model should answer directly (control group)
  - `hedging_reducible_uncertainty` (250 questions) — model should hedge or seek clarification
- **Adversarial red-team** — 191 adversarial queries designed to elicit overconfident responses

**Key metrics:**
- **Premature Closure rate (%)** — proportion of responses judged as committing prematurely (GPT-5.4 judge, verified against Claude Opus 4.7 judge; κ=0.685)
- **Rubric score** — physician-authored rubric score per response (normalized to % of max possible points)

---

### Prompting Conditions

1. **Baseline** — standard prompt, no abstention rules
2. **Safety** — explicit rules to pause, seek clarification, or abstain when unsure

---

### Models Evaluated

| Model | API |
|---|---|
| GPT-5.4 | Stanford AIHub / Azure OpenAI |
| Claude Opus 4.7 | Anthropic API |
| Grok 3 | xAI API |
| Gemini 2.5 Pro | Google Generative AI |
| DeepSeek R1 | DeepSeek API |

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
OPENAI_API_KEY=your_key          # or AIHUB_API_KEY for Stanford AIHub
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
GEMINI_API_KEY=your_key
XAI_API_KEY=your_key
```

---

## Running the Experiments

All scripts must be run from the **repo root**.

### 1. MedQA MCQ Experiment (500 Questions)

| File | Path |
|---|---|
| Questions | `data/questions.csv` |
| Answer key | `data/question_key.csv` |
| Raw results | `results_raw/` |
| Analysis script | `scripts/analyze/All_Model_Results.py` |
| Metrics output | `metrics/medqa_results.csv` |

```bash
# Run baseline and safety for each model
python scripts/gpt/gpt54_baseline_medqa.py
python scripts/gpt/gpt54_safety_medqa.py

python scripts/claude/claude46_baseline_medqa.py
python scripts/claude/claude46_safety_medqa.py

python scripts/grok3/grok3_baseline_medqa.py
python scripts/grok3/grok3_safety_medqa.py

python scripts/gemini/gemini_baseline.py
python scripts/gemini/gemini_safety.py

python scripts/deepseek/deepseek_baseline.py
python scripts/deepseek/deepseek_safety.py

# Analyze
python scripts/analyze/All_Model_Results.py
```

---

### 2. AfriMedQA MCQ Experiment (490 Questions)

| File | Path |
|---|---|
| Questions | `data/afrimedqa_questions.csv` |
| Answer key | `data/afrimedqa_questions_key.csv` |
| Raw results | `results_raw_afrimedqa/` |
| Analysis script | `scripts/analyze/Afrimedqa_all_model_results.py` |
| Metrics output | `metrics/afrimedqa_results.csv` |

```bash
python scripts/gpt/gpt54_baseline_afrimedqa.py
python scripts/gpt/gpt54_safety_afrimedqa.py

python scripts/claude/claude46_baseline_afrimedqa.py
python scripts/claude/claude46_safety_afrimedqa.py

python scripts/grok3/grok3_baseline_afrimedqa.py
python scripts/grok3/grok3_safety_afrimedqa.py

python scripts/gemini/gemini_afrimedqa.py   # runs baseline + safety

python scripts/deepseek/deepseek_afrimedqa.py  # runs baseline + safety

# Analyze
python scripts/analyze/Afrimedqa_all_model_results.py
```

---

### 3. HealthBench Experiment (861 Questions + 191 Adversarial)

#### 3a. Run the models

| File | Path |
|---|---|
| HB questions | `data/healthbench/healthbench_subset_combined.jsonl` |
| Red-team questions | `data/healthbench/healthbench_redteam.jsonl` |
| HB raw results | `results_raw_healthbench/` (`*_hb_hb.csv` files) |
| Red-team raw results | `results_raw_healthbench_redteam/` |

```bash
# HealthBench (861q) — baseline and safety for each model
python scripts/healthbench/runners/hb_baseline_gpt54_hb.py
python scripts/healthbench/runners/hb_safety_gpt54_hb.py

python scripts/healthbench/runners/hb_baseline_claudeopus_hb.py
python scripts/healthbench/runners/hb_safety_claudeopus_hb.py

python scripts/healthbench/runners/hb_baseline_grok3_hb.py
python scripts/healthbench/runners/hb_safety_grok3_hb.py

python scripts/healthbench/runners/hb_baseline_gemini_hb.py
python scripts/healthbench/runners/hb_safety_gemini_hb.py

python scripts/healthbench/runners/hb_baseline_deepseek_hb.py
python scripts/healthbench/runners/hb_safety_deepseek_hb.py

# Adversarial red-team (191q)
python scripts/healthbench/runners/hb_baseline_gpt54_redteam.py
python scripts/healthbench/runners/hb_safety_gpt54_redteam.py
# ... (same pattern for other models: claudeopus, grok3, gemini, deepseek)
```

#### 3b. Judge the responses (GPT-5.4 judge)

```bash
python scripts/healthbench/hb_judge_gpt.py
# Output → results_labeled_healthbench_new/ and results_labeled_healthbench_redteam_new/
```

#### 3c. Judge the responses (Claude Opus 4.7 judge — for agreement analysis)

```bash
python scripts/healthbench/hb_judge_claude.py
# Output → results_labeled_healthbench_claude_new/ and results_labeled_healthbench_redteam_claude_new/
```

#### 3d. Compute agreement, significance, and figures

```bash
python scripts/healthbench/hb_judge_agreement.py   # κ and r between judges
python scripts/healthbench/hb_significance.py       # McNemar + Wilcoxon tests
python scripts/healthbench/hb_analyze.py            # aggregate metrics → metrics/hb_*.csv
python scripts/healthbench/hb_figures.py            # figures → metrics/figures/
python scripts/healthbench/make_figures_revised.py  # revised scatter + summary table figures
```

---

## Repository Structure

```
NOTA-Benchmark/
├── data/
│   ├── questions.csv                        # MedQA 500-question dataset
│   ├── question_key.csv                     # MedQA answer key
│   ├── afrimedqa_questions.csv              # AfriMedQA dataset
│   ├── afrimedqa_questions_key.csv          # AfriMedQA answer key
│   └── healthbench/
│       ├── healthbench_subset_combined.jsonl  # HealthBench 861q subset
│       └── healthbench_redteam.jsonl          # Adversarial red-team 191q
├── scripts/
│   ├── gpt/            # GPT-5.4 (gpt54_baseline_*.py, gpt54_safety_*.py)
│   ├── claude/         # Claude Opus 4.7 (claude46_baseline_*.py, claude46_safety_*.py)
│   ├── grok3/          # Grok 3 (grok3_baseline_*.py, grok3_safety_*.py)
│   ├── gemini/         # Gemini 2.5 Pro (gemini_baseline.py, gemini_safety.py, gemini_afrimedqa.py)
│   ├── deepseek/       # DeepSeek R1 (deepseek_baseline.py, deepseek_safety.py, deepseek_afrimedqa.py)
│   ├── healthbench/
│   │   ├── runners/                         # Model run scripts (*_hb.py and *_redteam.py)
│   │   ├── hb_judge_gpt.py                  # GPT-5.4 judge (primary)
│   │   ├── hb_judge_claude.py               # Claude judge (agreement analysis)
│   │   ├── hb_judge_agreement.py            # Inter-rater agreement (κ, r)
│   │   ├── hb_significance.py               # McNemar + Wilcoxon tests
│   │   ├── hb_analyze.py                    # Aggregate metrics → metrics/hb_*.csv
│   │   ├── hb_figures.py                    # Standard figures
│   │   ├── make_figures_revised.py          # Revised scatter + summary table figures
│   │   ├── build_clinician_validation_sample.py    # Clinician validation sample batch 1
│   │   ├── build_clinician_validation_sample_v2.py # Clinician validation sample batch 2
│   │   └── make_clinician_review_workbook.py       # Blinded Excel workbook for clinician review
│   └── analyze/
│       ├── All_Model_Results.py             # MedQA metrics
│       └── Afrimedqa_all_model_results.py   # AfriMedQA metrics
├── results_raw/                             # Generated by running model scripts (not in repo)
├── results_raw_afrimedqa/                   # Generated by running model scripts (not in repo)
├── results_raw_healthbench/                 # Generated by running model scripts (not in repo)
├── results_raw_healthbench_redteam/         # Generated by running model scripts (not in repo)
├── results_labeled_healthbench_new/         # Generated by hb_judge_gpt.py (not in repo)
├── results_labeled_healthbench_claude_new/  # Generated by hb_judge_claude.py (not in repo)
├── results_labeled_healthbench_redteam_new/        # Generated by hb_judge_gpt.py (not in repo)
├── results_labeled_healthbench_redteam_claude_new/ # Generated by hb_judge_claude.py (not in repo)
├── metrics/
│   ├── medqa_results.csv
│   ├── afrimedqa_results.csv
│   ├── hb_aggregate.csv
│   ├── hb_aggregate_overall.csv
│   ├── hb_redteam_aggregate.csv
│   ├── hb_significance.csv
│   ├── hb_judge_agreement.csv
│   └── figures/                             # All output figures
├── environment.yml
├── LICENSE
└── README.md
```

---

## Authors

Rebecca Handler, Suhana Bedi, Nigam H. Shah  
Stanford University  
[handler9@stanford.edu](mailto:handler9@stanford.edu)
