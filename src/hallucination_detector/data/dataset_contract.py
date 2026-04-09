"""
Контракт обучающего CSV: текстовые пары + метка + прослеживаемость (id, source).
Без колонок prompt/response обучение не запускается — нельзя подменить train «анонимными» числами.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_TEXT_COLUMNS = ("prompt", "response", "label")
REQUIRED_META = ("id", "source")


def _norm_pair(prompt: str, response: str) -> tuple[str, str]:
    p = " ".join(str(prompt).split()).strip().lower()
    r = " ".join(str(response).split()).strip().lower()
    return p, r


def validate_train_csv(
    path: Path | str,
    *,
    chunk_size: int = 100_000,
    strict_meta: bool = True,
) -> None:
    """
    Raises ValueError если CSV не является текст-first датасетом.

    Проверяется весь файл по чанкам на пустые prompt/response.
    """
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Файл не найден: {path}")

    head = pd.read_csv(path, nrows=1)
    cols = set(head.columns)

    for c in REQUIRED_TEXT_COLUMNS:
        if c not in cols:
            raise ValueError(
                f"Train CSV должен содержать колонку {c!r}. "
                f"Обучение только на числовых фичах без исходных текстов запрещено контрактом. "
                f"Колонки: {sorted(cols)}"
            )

    if strict_meta:
        for c in REQUIRED_META:
            if c not in cols:
                raise ValueError(
                    f"Ожидается колонка {c!r} (прослеживаемость источника строки). "
                    f"Колонки: {sorted(cols)}"
                )

    total = 0
    bad = 0
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        empty_p = chunk["prompt"].astype(str).str.strip() == ""
        empty_r = chunk["response"].astype(str).str.strip() == ""
        bad += int((empty_p | empty_r).sum())
        total += len(chunk)

    if total == 0:
        raise ValueError("Train CSV не содержит ни одной строки данных.")

    if bad:
        raise ValueError(
            f"Найдены пустые prompt/response: {bad} из {total} строк. "
            "Исправьте датасет или препроцессинг."
        )


def count_pair_overlap(
    train_csv: Path | str,
    other_csv: Path | str,
    *,
    other_prompt_col: str = "prompt",
    other_response_col: str = "response",
) -> int:
    """
    Количество пересечений по нормализованным парам (prompt, response)
    между train_csv и other_csv.
    """
    train_csv = Path(train_csv)
    other_csv = Path(other_csv)
    if not train_csv.is_file() or not other_csv.is_file():
        return 0

    tdf = pd.read_csv(train_csv, usecols=["prompt", "response"])
    odf = pd.read_csv(other_csv)
    if other_prompt_col not in odf.columns or other_response_col not in odf.columns:
        return 0

    train_set = {_norm_pair(p, r) for p, r in zip(tdf["prompt"], tdf["response"])}
    other_set = {
        _norm_pair(p, r) for p, r in zip(odf[other_prompt_col], odf[other_response_col])
    }
    return len(train_set & other_set)
