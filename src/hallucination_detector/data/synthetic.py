"""Синтетические prompt–response с эвристической разметкой (без внешних API)."""

from __future__ import annotations

import random
from typing import Any

from hallucination_detector.data.source_tags import SYNTHETIC


def generate_synthetic_rows(n: int, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pool = [
        ("Сколько будет 2 + 2?", "4", 0),
        ("Сколько будет 2 + 2?", "5", 1),
        ("Сколько будет 2 + 2?", "22", 1),
        ("What is 7 * 8?", "56", 0),
        ("What is 7 * 8?", "54", 1),
        ("Столица Франции?", "Париж", 0),
        ("Столица Франции?", "Берлин", 1),
        ("Capital of Japan?", "Tokyo", 0),
        ("Capital of Japan?", "Seoul", 1),
        ("Что такое вода?", "Вода — это H2O.", 0),
        ("Что такое вода?", "Вода — это металл, плавящийся при 2000°C.", 1),
    ]
    for i in range(n):
        p, r, lab = rng.choice(pool)
        rows.append(
            {
                "prompt": f"{p} [syn:{i}]",
                "response": r,
                "label": lab,
                "source": SYNTHETIC,
            }
        )
    return rows
