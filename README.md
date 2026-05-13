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
  - `underspecified_low_risk` (180 questions) — model should seek clarification
  - `underspecified_urgent` (195 questions) — model should escalate or ask before advising
  - `sufficient_context` (241 questions) — model should answer directly (control group)
  - `hedging_reducible_uncertainty` (245 questions) — model should hedge or seek clarification
- **Adversarial red-team** — 191 adversarial queries designed to elicit overconfident responses

**Key metrics:**
- **Premature Closure rate (%)** — proportion of responses judged as committing prematurely (GPT-5.4 judge, verified against Claude Opus 4.7 judge; κ=0.685)
- **Rubric score** — physician-authored rubric score per response (−10 to +10 pts, can be negative)

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
| Analysis script | `scripts/analyze-results/All_Model_Results.py` |
| Metrics output | `metrics/medqa_results_new.csv` |

```bash
# Run baseline and safety for each model
python scripts/gpt-tests/gpt54_baseline_medqa.py
python scripts/gpt-tests/gpt54_safety_medqa.py

python scripts/claude-tests/claude46_baseline_medqa.py
python scripts/claude-tests/claude46_safety_medqa.py

python scripts/grok3-tests/grok3_baseline_medqa.py
python scripts/grok3-tests/grok3_safety_medqa.py

python scripts/gemini-scripts/gemini_baseline.py
python scripts/gemini-scripts/gemini_safety.py

python scripts/deepseek-tests/deepseek_baseline.py
python scripts/deepseek-tests/deepseek_safety.py

# Analyze
python scripts/analyze-results/All_Model_Results.py
```

---

### 2. AfriMedQA MCQ Experiment (490 Questions)

| File | Path |
|---|---|
| Questions | `data/afrimedqa_questions.csv` |
| Answer key | `data/afrimedqa_questions_key.csv` |
| Raw results | `results_raw_afrimedqa/` |
| Analysis script | `scripts/analyze-results/Afrimedqa_all_model_results.py` |
| Metrics output | `metrics/afrimedqa_results_new.csv` |

```bash
python scripts/gpt-tests/gpt54_baseline_afrimedqa.py
python scripts/gpt-tests/gpt54_safety_afrimedqa.py

python scripts/claude-tests/claude46_baseline_afrimedqa.py
python scripts/claude-tests/claude46_safety_afrimedqa.py

python scripts/grok3-tests/grok3_baseline_afrimedqa.py
python scripts/grok3-tests/grok3_safety_afrimedqa.py

python scripts/gemini-scripts/gemini_afrimedqa.py   # runs baseline + safety

python scripts/deepseek-tests/deepseek_afrimedqa.py  # runs baseline + safety

# Analyze
python scripts/analyze-results/Afrimedqa_all_model_results.py
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
python scripts/healthbench/hb_judge_gpt_new.py
# Output → results_labeled_healthbench_new/ and results_labeled_healthbench_redteam_new/
```

#### 3c. Judge the responses (Claude Opus 4.7 judge — for agreement analysis)

```bash
python scripts/healthbench/hb_judge_claude_new.py
# Output → results_labeled_healthbench_claude_new/ and results_labeled_healthbench_redteam_claude_new/
```

#### 3d. Compute agreement, significance, and figures

```bash
python scripts/healthbench/hb_judge_agreement_new.py   # κ and r between judges
python scripts/healthbench/hb_significance_new.py       # McNemar + Wilcoxon tests
python scripts/healthbench/hb_analyze_new.py            # aggregate metrics → metrics/hb_*.csv
python scripts/healthbench/hb_figures_new.py            # figures → metrics/figures/
python scripts/healthbench/make_figures_revised.py      # revised scatter + summary table figures
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
│       ├── healthbench_subset_combined.jsonl  # HealthBench 861q
│       └── healthbench_redteam.jsonl          # Adversarial red-team 191q
├── scripts/
│   ├── gpt-tests/          # GPT-5.4 (gpt54_baseline_*.py, gpt54_safety_*.py)
│   ├── claude-tests/       # Claude Opus 4.7 (claude46_baseline_*.py, claude46_safety_*.py)
│   ├── grok3-tests/        # Grok 3 (grok3_baseline_*.py, grok3_safety_*.py)
│   ├── gemini-scripts/     # Gemini 2.5 Pro (gemini_baseline.py, gemini_safety.py, gemini_afrimedqa.py)
│   ├── deepseek-tests/     # DeepSeek R1 (deepseek_baseline.py, deepseek_safety.py, deepseek_afrimedqa.py)
│   ├── healthbench/
│   │   ├── runners/                          # Model run scripts (*_hb.py and *_redteam.py)
│   │   ├── hb_judge_gpt_new.py
│   │   ├── hb_judge_claude_new.py
│   │   ├── hb_judge_agreement_new.py
│   │   ├── hb_significance_new.py
│   │   ├── hb_analyze_new.py
│   │   ├── hb_figures_new.py
│   │   ├── make_figures_revised.py           # Revised scatter + summary table figures
│   │   ├── build_clinician_validation_sample.py    # Clinician validation sample batch 1
│   │   ├── build_clinician_validation_sample_v2.py # Clinician validation sample batch 2
│   │   └── make_clinician_review_workbook.py       # Formats blinded Excel workbook for clinician review
│   └── analyze-results/
│       ├── All_Model_Results.py             # MedQA metrics
│       └── Afrimedqa_all_model_results.py   # AfriMedQA metrics
├── results_raw/                             # Generated by running model scripts (not in repo)
├── results_raw_afrimedqa/                   # Generated by running model scripts (not in repo)
├── results_raw_healthbench/                 # Generated by running model scripts (not in repo)
├── results_raw_healthbench_redteam/         # Generated by running model scripts (not in repo)
├── results_labeled_healthbench_new/         # Generated by hb_judge_gpt_new.py (not in repo)
├── results_labeled_healthbench_claude_new/  # Generated by hb_judge_claude_new.py (not in repo)
├── results_labeled_healthbench_redteam_new/         # Generated by hb_judge_gpt_new.py (not in repo)
├── results_labeled_healthbench_redteam_claude_new/  # Generated by hb_judge_claude_new.py (not in repo)
├── metrics/
│   ├── medqa_results_new.csv
│   ├── afrimedqa_results_new.csv
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
