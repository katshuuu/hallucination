from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional

import pandas as pd

from hallucination_detector.columns_util import response_column
from hallucination_detector.pipeline import HallucinationDetector, load_detector_from_config
from hallucination_detector.settings import load_config, repo_root


def score_dataframe_cross_encoder(
    df: pd.DataFrame,
    detector: HallucinationDetector,
    prompt_col: str,
    response_col: str,
) -> pd.Series:
    pairs = list(zip(df[prompt_col].astype(str), df[response_col].astype(str)))
    probs = detector.predict_proba(pairs)
    return pd.Series(probs, index=df.index, name="hallucination_prob")


def run_score(
    config_path: Path,
    input_csv: Path,
    output_csv: Path,
    *,
    model_override: Optional[str] = None,
) -> None:
    cfg = load_config(config_path)
    cols = cfg["columns"]
    prompt_col = cols["prompt"]
    backend = cfg.get("backend", "tabular")

    if model_override:
        cfg = copy.deepcopy(cfg)
        if backend == "tabular":
            cfg["paths"] = dict(cfg["paths"])
            cfg["paths"]["artifact_dir"] = model_override
        else:
            cfg["model"] = dict(cfg["model"])
            cfg["model"]["name"] = model_override

    df = pd.read_csv(input_csv)
    if prompt_col not in df.columns:
        raise ValueError(f"CSV must contain prompt column {prompt_col!r}; got {list(df.columns)}")
    resp_col = response_column(df, cols)
    df = df.copy()

    if backend == "tabular":
        from hallucination_detector.tabular.scorer import load_scorer_from_config, score_dataframe_tabular

        scorer = load_scorer_from_config(cfg)
        df["hallucination_prob"] = score_dataframe_tabular(df, scorer, cols)
    elif backend == "cross_encoder":
        detector = load_detector_from_config(cfg)
        df["hallucination_prob"] = score_dataframe_cross_encoder(df, detector, prompt_col, resp_col)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} rows to {output_csv}")


def main() -> None:
    p = argparse.ArgumentParser(description="Score CSV with hallucination probabilities")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="tabular: каталог с tabular_bundle.joblib; cross_encoder: HF id или путь",
    )
    args = p.parse_args()
    run_score(args.config, args.input, args.output, model_override=args.model)


if __name__ == "__main__":
    main()
