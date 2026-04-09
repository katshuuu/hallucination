#!/usr/bin/env python3
"""
Проверка, что train CSV не содержит пар из public benchmark.
"""
from __future__ import annotations

from pathlib import Path

from hallucination_detector.data.dataset_contract import count_pair_overlap
from hallucination_detector.settings import repo_root


def main() -> int:
    root = repo_root()
    train_csv = root / "data" / "processed" / "train_merged.csv"
    public_csv = root / "knowledge_bench_public.csv"
    if not train_csv.is_file():
        print(f"[skip] Нет {train_csv}")
        return 0
    if not public_csv.is_file():
        print(f"[skip] Нет {public_csv}")
        return 0

    overlap = count_pair_overlap(
        train_csv,
        public_csv,
        other_prompt_col="prompt",
        other_response_col="model_answer",
    )
    if overlap > 0:
        print(f"[FAIL] Найдено пересечений train vs public benchmark: {overlap}")
        return 2
    print("[OK] Пересечений train vs public benchmark не найдено")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

