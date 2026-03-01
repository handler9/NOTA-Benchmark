# LLM Clinical Safety NOTA Benchmark

This repository contains an evaluation framework for measuring **clinical safety**, **abstention behavior**, and **false-action risk** in large language models (LLMs). The benchmark uses a mixed dataset of INTACT and TRUE-NOTA medical questions to evaluate safe decision-making and refusal behavior.

---

## Benchmark Overview

The benchmark includes:

- **INTACT questions** — one correct answer present  
- **TRUE-NOTA questions** — correct answer removed (the safest action is to abstain)

All questions are **shuffled together**, and INTACT questions have one distractor removed so all items contain **exactly four answer choices**.

This benchmark measures:

- False Action Rate (answering when abstention is correct)  
- Safe Abstention Rate on TRUE-NOTA questions  
- Incorrect Abstention Rate on INTACT questions  
- Confidence patterns and confidence gaps  
- Model behavior across multiple prompting strategies  

---

## Datasets

Located in the `data/` folder:

- **question_key.csv**  
  Contains `question_type` (INTACT vs TRUE-NOTA) and correct answers for INTACT questions.

- **questions_mixed_intact_true_nota.csv**  
  The exact set of **500 shuffled questions** (250 INTACT + 250 TRUE-NOTA).

---

## Models Evaluated

This repository currently includes runs for:

- GPT-5  
- Claude  
- DeepSeek  
- Llama  

Each model is evaluated under multiple prompting conditions.

---

## Prompting Conditions

Models are tested under four prompt styles:

1. **Baseline Prompt**  
   No abstention rules.

2. **Safety Prompt**  
   Explicit rules to abstain when unsure or unsafe.

3. **Think-Then-Decide Prompt**  
   Internal reasoning before answering.

4. **Answer-Then-Double-Check Prompt**  
   Model answers first, then performs a safety review.

---

## Raw Model Outputs

All raw results are saved in the `results_raw/` folder.

Files follow this naming pattern:

```
promptX-condition-model.csv
```

Examples:

- `prompt1-baseline-claude.csv`  
- `prompt2-safety-gpt5.csv`  
- `prompt3-think-deepseek.csv`  
- `prompt4-doublecheck-llama.csv`

Each CSV includes:

- Question text (`question_id`, `stem`, `option_A`–`option_D`)  
- Model outputs (`*_raw`, `*_choice`, `*_abstain_code`, `*_confidence`, `*_rationale`)  

---

## Analysis Script

Main script:

```
scripts/results.py
```

Outputs:

- Accuracy on INTACT  
- False Action Rate (TRUE-NOTA)  
- Abstain Rate on TRUE-NOTA  
- Abstain Rate on INTACT  
- Confidence metrics  
- Confidence gap  
- Abstained-on-INTACT count (new metric)

Results are written to:

```
metrics/updatedresults2.csv
```

---

## Study Design Summary

### Question Setup

- 250 INTACT questions  
- 250 TRUE-NOTA questions  
- Combined and shuffled into: `questions_mixed_intact_true_nota.csv`  
- Metadata in: `question_key.csv`

### Prompts Tested

- Baseline  
- Safety  
- Think-then-decide  
- Answer-then-double-check  

### Key Metrics

- False Action Rate  
- Safe Abstention (TRUE-NOTA)  
- Incorrect Abstention (INTACT)  
- Confidence Gap  

---

## Installation & Environment Setup

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate nota-benchmark
```

To update:

```bash
conda env update -f environment.yml --prune
```

### 2. Optional: Install Additional Packages

```bash
pip install -r requirements.txt
```

(Only if you add a requirements.txt.)

---

## Running the Analysis

1. Ensure folder structure:

```
LLM-NOTA-Benchmark/
  data/
  results_raw/
  metrics/
  scripts/
  environment.yml
  README.md
```

2. Activate environment:

```bash
conda activate nota-benchmark
```

3. Run:

```bash
python scripts/results.py
```

4. View:

```
metrics/updatedresults2.csv
```

---

## Generating New Model Outputs

### 1. Activate environment

```bash
conda activate nota-benchmark
```

### 2. (Optional) Add API keys

Create `.env`:

```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

### 3. Run a model script

```bash
python scripts/run_prompt1_baseline_claude.py
```

### 4. Check output

```
results_raw/prompt1-baseline-claude.csv
```

### 5. Re-run analysis

```bash
python scripts/results.py
```

---

## Repository Structure

```
LLM-NOTA-Benchmark/
├── data/
│   ├── question_key.csv
│   └── questions_mixed_intact_true_nota.csv
│
├── results_raw/
│   └── promptX-condition-model.csv
│
├── metrics/
│   └── updatedresults2.csv
│
├── scripts/
│   ├── model generation scripts
│   └── results.py
│
├── environment.yml
└── README.md
```
