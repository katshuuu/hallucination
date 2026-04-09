from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hallucination_detector.columns_util import response_column
from hallucination_detector.settings import load_config


class TabularScorer:
    """Загрузка bundle (extractor + LGBM) и скоринг датафрейма."""

    def __init__(self, bundle: dict[str, Any]) -> None:
        self.extractor: Any = bundle["extractor"]
        self.classifier: Any = bundle["classifier"]
        self.feature_names: list[str] = list(bundle.get("feature_names", []))

    def predict_proba_positive(self, df: pd.DataFrame, prompt_col: str, response_col: str) -> np.ndarray:
        X = self.extractor.transform(df, prompt_col, response_col)
        names = self.feature_names
        if names and X.shape[1] == len(names):
            x_df = pd.DataFrame.sparse.from_spmatrix(X, columns=names)
            proba = self.classifier.predict_proba(x_df)
        else:
            proba = self.classifier.predict_proba(X)
        return proba[:, 1]

    def score_one(self, prompt: str, response: str, prompt_col: str = "prompt", response_col: str = "response") -> float:
        df = pd.DataFrame([{prompt_col: prompt, response_col: response}])
        return float(self.predict_proba_positive(df, prompt_col, response_col)[0])


def load_tabular_scorer(artifact_dir: Path | str) -> TabularScorer:
    path = Path(artifact_dir) / "tabular_bundle.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Нет артефакта {path}. Сначала ./scripts/train.sh")
    bundle = joblib.load(path)
    return TabularScorer(bundle)


def load_scorer_from_config(cfg: dict) -> TabularScorer:
    paths = cfg["paths"]
    return load_tabular_scorer(paths["artifact_dir"])


def score_dataframe_tabular(
    df: pd.DataFrame,
    scorer: TabularScorer,
    cols: dict,
) -> pd.Series:
    prompt_col = cols["prompt"]
    resp_col = response_column(df, cols)
    p = scorer.predict_proba_positive(df, prompt_col, resp_col)
    return pd.Series(p, index=df.index, name="hallucination_prob")
