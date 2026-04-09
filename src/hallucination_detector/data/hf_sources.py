"""Загрузка открытых датасетов с Hugging Face (только для обучения, не для инференса)."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from hallucination_detector.data.normalize import clean_text
from hallucination_detector.data.source_tags import OPEN_DATASET


def load_truthful_qa(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "generation", split="train")
    rows: list[dict[str, Any]] = []
    for row in tqdm(ds, desc="truthful_qa"):
        if len(rows) >= limit:
            break
        q = clean_text(row["question"])
        best = row.get("best_answer")
        if best:
            rows.append(
                {
                    "prompt": q,
                    "response": clean_text(best),
                    "label": 0,
                    "source": OPEN_DATASET,
                }
            )
        incorrect = row.get("incorrect_answers") or []
        for w in incorrect[:3]:
            if len(rows) >= limit:
                break
            rows.append(
                {
                    "prompt": q,
                    "response": clean_text(w),
                    "label": 1,
                    "source": OPEN_DATASET,
                }
            )
    return rows[:limit]


def load_snli(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("snli", split="train")
    rows: list[dict[str, Any]] = []
    # 0 entailment, 1 neutral, 2 contradiction
    for row in tqdm(ds, desc="snli"):
        if len(rows) >= limit:
            break
        if row["label"] == -1:
            continue
        premise = clean_text(row["premise"])
        hyp = clean_text(row["hypothesis"])
        lab = 1 if int(row["label"]) == 2 else 0
        rows.append(
            {
                "prompt": premise,
                "response": hyp,
                "label": lab,
                "source": OPEN_DATASET,
            }
        )
    return rows[:limit]


def load_paws(limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("paws", "labeled_final", split="train")
    rows: list[dict[str, Any]] = []
    for row in tqdm(ds, desc="paws"):
        if len(rows) >= limit:
            break
        s1 = clean_text(row["sentence1"])
        s2 = clean_text(row["sentence2"])
        # 1 = paraphrase (согласованная пара), 0 = не парафраза
        is_para = int(row["label"]) == 1
        rows.append(
            {
                "prompt": s1,
                "response": s2,
                "label": 0 if is_para else 1,
                "source": OPEN_DATASET,
            }
        )
    return rows[:limit]
