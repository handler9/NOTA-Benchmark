from __future__ import annotations

import random
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "metrics"
SEED = 20260506

MODELS = {
    "gpt54": "GPT-5.4",
    "claudeopus": "Claude Opus 4.7",
    "grok3": "Grok 3",
    "gemini": "Gemini 2.5 Pro",
    "deepseek": "DeepSeek R1",
}
CONDITIONS = ["baseline", "safety"]
STANDARD_SUBSETS = [
    "underspecified_low_risk",
    "underspecified_urgent",
    "sufficient_context",
    "hedging_reducible_uncertainty",
]


NON_ENGLISH_MARKERS = {
    "ich",
    "welche",
    "necesito",
    "paciente",
    "acontece",
    "posso",
    "gerenciar",
    "qu est",
    "c est",
    "são",
    "soy",
    "clínica",
    "atención",
    "mon frère",
    "je ne",
    "peux tu",
    "quels",
    "quiero",
    "puede decir",
    "cuál",
    "quanto",
    "café",
    "bebê",
    "entonces",
    "seguro decir",
    "olá",
    "roteiro",
    "é esperado",
    "qué puedo",
    "chat se",
    "tô",
    "que significa",
    "sintoma",
    "entumecimiento",
    "et si",
    "tam emin",
    "doktora",
    "gecmiyor",
    "abhi",
    "bahut",
    "doctor vaghaira",
    "un mio",
    "paziente",
    "cardiopatia",
    "instabilità",
    "referto",
    "mandarin",
    "translate my plan",
}


def has_non_latin_script(text: str) -> bool:
    return bool(
        re.search(
            r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
            r"\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
            r"\u0400-\u04ff\u1200-\u137f]",
            text,
        )
    )


def is_english_question(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", str(text).lower()).strip()
    if has_non_latin_script(cleaned):
        return False
    if any(marker in cleaned for marker in NON_ENGLISH_MARKERS):
        return False
    return True


def read_labeled(dataset: str, judge: str, model_key: str, condition: str) -> pd.DataFrame:
    if dataset == "HealthBench":
        folder = "results_labeled_healthbench_new" if judge == "gpt" else "results_labeled_healthbench_claude_new"
        filename = f"{model_key}_{condition}_hb_hb_labeled.csv"
    else:
        folder = (
            "results_labeled_healthbench_redteam_new"
            if judge == "gpt"
            else "results_labeled_healthbench_redteam_claude_new"
        )
        filename = f"{model_key}_{condition}_redteam_hb_labeled.csv"
    return pd.read_csv(ROOT / folder / filename)


def build_candidates(dataset: str) -> pd.DataFrame:
    rows = []
    for model_key, model_name in MODELS.items():
        for condition in CONDITIONS:
            gpt = read_labeled(dataset, "gpt", model_key, condition)
            claude = read_labeled(dataset, "claude", model_key, condition)
            raw_col = f"{model_key}_{condition}_raw"
            behavior_col = f"{model_key}_{condition}_behavior"
            pc_col = f"{model_key}_{condition}_premature_closure"
            earned_col = f"{model_key}_{condition}_score_earned"
            possible_col = f"{model_key}_{condition}_score_possible"
            reason_col = f"{model_key}_{condition}_reasoning"

            keep = [
                "prompt_id",
                "question",
                "full_conversation",
                "subset",
                raw_col,
                behavior_col,
                pc_col,
                earned_col,
                possible_col,
                reason_col,
            ]
            gpt = gpt[keep].rename(
                columns={
                    raw_col: "model_response",
                    behavior_col: "gpt_judge_behavior",
                    pc_col: "gpt_judge_premature_closure",
                    earned_col: "score_earned",
                    possible_col: "score_possible",
                    reason_col: "gpt_judge_reasoning",
                }
            )
            claude = claude[["prompt_id", behavior_col, pc_col, reason_col]].rename(
                columns={
                    behavior_col: "claude_judge_behavior",
                    pc_col: "claude_judge_premature_closure",
                    reason_col: "claude_judge_reasoning",
                }
            )
            merged = gpt.merge(claude, on="prompt_id", how="left")
            merged["dataset"] = dataset
            merged["model_key"] = model_key
            merged["model"] = model_name
            merged["condition"] = condition
            rows.append(merged)

    out = pd.concat(rows, ignore_index=True)
    out = out[out["question"].map(is_english_question)].copy()
    out = out[
        out["gpt_judge_premature_closure"].isin(["yes", "no"])
        & out["claude_judge_premature_closure"].isin(["yes", "no"])
    ].copy()
    out["score_pct"] = out["score_earned"] / out["score_possible"]
    return out


def sample_standard(candidates: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED)
    selected = []
    used_prompts = set()
    strata = [(m, c) for m in MODELS for c in CONDITIONS]

    for subset_i, subset in enumerate(STANDARD_SUBSETS):
        targets = {stratum: 2 for stratum in strata}
        offset = 0 if subset_i % 2 == 0 else 5
        for stratum in strata[offset : offset + 5]:
            targets[stratum] += 1

        for model_key, condition in strata:
            pool = candidates[
                (candidates["subset"] == subset)
                & (candidates["model_key"] == model_key)
                & (candidates["condition"] == condition)
                & (~candidates["prompt_id"].isin(used_prompts))
            ].copy()
            pool = pool.sample(frac=1, random_state=rng.randint(0, 10**9))
            take = pool.head(targets[(model_key, condition)])
            selected.append(take)
            used_prompts.update(take["prompt_id"].tolist())

    sample = pd.concat(selected, ignore_index=True)
    if len(sample) != 100:
        raise RuntimeError(f"Expected 100 standard rows, got {len(sample)}")
    return sample


def sample_redteam(candidates: pd.DataFrame) -> pd.DataFrame:
    rng = random.Random(SEED + 1)
    selected = []
    used_prompts = set()

    for model_key in MODELS:
        for condition in CONDITIONS:
            pool = candidates[
                (candidates["model_key"] == model_key)
                & (candidates["condition"] == condition)
                & (~candidates["prompt_id"].isin(used_prompts))
            ].copy()
            pool = pool.sample(frac=1, random_state=rng.randint(0, 10**9))
            take = pool.head(5)
            selected.append(take)
            used_prompts.update(take["prompt_id"].tolist())

    sample = pd.concat(selected, ignore_index=True)
    if len(sample) != 50:
        raise RuntimeError(f"Expected 50 red-team rows, got {len(sample)}")
    return sample


def main() -> None:
    standard = sample_standard(build_candidates("HealthBench"))
    redteam = sample_redteam(build_candidates("HealthBench Professional red-team"))
    sample = pd.concat([standard, redteam], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SEED).reset_index(drop=True)
    sample.insert(0, "validation_id", [f"CV-{i:03d}" for i in range(1, len(sample) + 1)])

    clinician_cols = {
        "clinician1_premature_closure": "",
        "clinician1_behavior": "",
        "clinician1_notes": "",
        "clinician2_premature_closure": "",
        "clinician2_behavior": "",
        "clinician2_notes": "",
        "adjudicated_premature_closure": "",
        "adjudication_notes": "",
    }
    for col, value in clinician_cols.items():
        sample[col] = value

    key_cols = [
        "validation_id",
        "dataset",
        "model",
        "model_key",
        "condition",
        "subset",
        "prompt_id",
        "question",
        "full_conversation",
        "model_response",
        "gpt_judge_behavior",
        "gpt_judge_premature_closure",
        "gpt_judge_reasoning",
        "claude_judge_behavior",
        "claude_judge_premature_closure",
        "claude_judge_reasoning",
        "score_earned",
        "score_possible",
        "score_pct",
        *clinician_cols.keys(),
    ]
    blinded_cols = [
        "validation_id",
        "dataset",
        "subset",
        "question",
        "full_conversation",
        "model_response",
        *clinician_cols.keys(),
    ]

    key_path = OUT_DIR / "clinician_validation_sample_150_key.csv"
    blinded_path = OUT_DIR / "clinician_validation_sample_150_blinded.csv"
    sample[key_cols].to_csv(key_path, index=False)
    sample[blinded_cols].to_csv(blinded_path, index=False)

    print(f"Wrote {key_path}")
    print(f"Wrote {blinded_path}")
    print("\nCounts by dataset:")
    print(sample["dataset"].value_counts().to_string())
    print("\nCounts by subset:")
    print(sample["subset"].value_counts().to_string())
    print("\nCounts by model/condition:")
    print(sample.groupby(["model", "condition"]).size().to_string())
    print("\nJudge PC counts:")
    print(sample["gpt_judge_premature_closure"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
