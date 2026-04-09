"""Табличный стек: признаки + LightGBM/CatBoost (без end-to-end LLM на инференсе)."""

from hallucination_detector.tabular.scorer import TabularScorer, load_tabular_scorer

__all__ = ["TabularScorer", "load_tabular_scorer"]
