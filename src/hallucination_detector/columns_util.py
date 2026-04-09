from __future__ import annotations

import pandas as pd


def response_column(df: pd.DataFrame, cols_cfg: dict) -> str:
    """Имя колонки ответа: response > model_answer > answer."""
    for key in ("response", "model_answer", "answer"):
        if key in cols_cfg and cols_cfg[key] in df.columns:
            return cols_cfg[key]
    for key in ("response", "model_answer", "answer"):
        if key in df.columns:
            return key
    raise ValueError(
        f"Нет колонки ответа (ожидались response/model_answer/answer). Колонки: {list(df.columns)}"
    )
