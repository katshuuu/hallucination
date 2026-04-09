"""Нормализация поля source к фиксированному перечислению для честности и анализа."""

from __future__ import annotations

# Допустимые значения в финальном train CSV
OPEN_DATASET = "open_dataset"
SYNTHETIC = "synthetic"
JUDGE_LABELED = "judge_labeled"
MANUAL = "manual"
SUPPLEMENTAL = "supplemental"


def normalize_source(raw: str) -> str:
    s = str(raw).strip().lower()
    if s in (OPEN_DATASET, SYNTHETIC, JUDGE_LABELED, MANUAL, SUPPLEMENTAL):
        return s
    if "judge" in s or "gpt" in s or "llm" in s:
        return JUDGE_LABELED
    if "supplemental" in s or "extra_qa" in s:
        return SUPPLEMENTAL
    if "manual" in s or s == "manual_validation":
        return MANUAL
    if "synthetic" in s or s.startswith("syn"):
        return SYNTHETIC
    # truthful_qa, snli, paws, hf, open, dataset
    if any(
        x in s
        for x in (
            "truthful",
            "snli",
            "paws",
            "open",
            "hf",
            "dataset",
            "qa_best",
            "incorrect",
            "paraphrase",
            "contradiction",
            "entail",
        )
    ):
        return OPEN_DATASET
    return OPEN_DATASET
