"""
Минимальный веб-интерфейс: загрузка CSV, батч-скоринг, метрики качества (при наличии label).
"""

from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from sklearn.metrics import accuracy_score, average_precision_score

from hallucination_detector.columns_util import response_column
from hallucination_detector.detection_timing import build_timing_result
from hallucination_detector.tabular.scorer import TabularScorer, score_dataframe_tabular

logger = logging.getLogger("hallucination_detector.api.ui")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UI_ROWS = 10_000

GetTabularState = Callable[
    [],
    Tuple[Optional[TabularScorer], Optional[Dict[str, Any]], Optional[Dict[str, Any]]],
]


def create_ui_router(get_state: GetTabularState) -> APIRouter:
    router = APIRouter(tags=["ui"])

    @router.get("/", include_in_schema=False)
    def root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui", status_code=307)

    @router.get("/ui", include_in_schema=False)
    def ui_page() -> FileResponse:
        path = STATIC_DIR / "index.html"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="UI not found (missing static/index.html)")
        return FileResponse(path, media_type="text/html; charset=utf-8")

    @router.post("/api/ui/score-csv")
    async def score_uploaded_csv(file: UploadFile = File(...)) -> dict[str, Any]:
        scorer, cols, run_cfg = get_state()
        if scorer is None or cols is None or run_cfg is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Ожидается файл .csv")

        raw = await file.read()
        if len(raw) > 50 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Файл слишком большой (макс. 50 МБ)")

        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {e}") from e

        if len(df) == 0:
            raise HTTPException(status_code=400, detail="CSV пустой")
        if len(df) > MAX_UI_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"Слишком много строк ({len(df)}). Максимум {MAX_UI_ROWS} для UI.",
            )

        prompt_col = cols["prompt"]
        if prompt_col not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Нет колонки «{prompt_col}». Найдены: {list(df.columns)}",
            )
        try:
            resp_col = response_column(df, cols)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        t0 = time.perf_counter()
        try:
            probs = score_dataframe_tabular(df, scorer, cols)
        except Exception as e:
            logger.exception("ui score failed")
            raise HTTPException(status_code=500, detail=f"Ошибка скоринга: {e}") from e
        wall_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_result(wall_ms, run_cfg)

        out_df = df.copy()
        out_df["hallucination_prob"] = probs.values

        metrics: dict[str, Any] = {
            "num_rows": int(len(df)),
            "mean_hallucination_prob": float(probs.mean()),
            "min_hallucination_prob": float(probs.min()),
            "max_hallucination_prob": float(probs.max()),
            "fraction_above_0_5": float((probs >= 0.5).mean()),
            "wall_time_ms_total": round(timing.wall_time_ms, 3),
            "billable_time_ms_total": round(timing.billable_time_ms, 3),
            "ms_per_row_wall": round(timing.wall_time_ms / max(len(df), 1), 4),
            "ms_per_row_billable": round(timing.billable_time_ms / max(len(df), 1), 4),
            "jury_max_billable_ms": timing.jury_max_billable_ms,
        }

        label_col = cols.get("label", "label")
        if label_col in df.columns:
            y = pd.to_numeric(df[label_col], errors="coerce")
            valid = y.notna() & y.isin([0, 1])
            n_valid = int(valid.sum())
            metrics["labeled_rows_used"] = n_valid
            if n_valid >= 2 and y[valid].nunique() >= 2:
                yv = y[valid].astype(int).values
                pv = probs[valid].values
                metrics["pr_auc"] = float(average_precision_score(yv, pv))
                pred_cls = (pv >= 0.5).astype(int)
                metrics["accuracy_at_threshold_0_5"] = float(accuracy_score(yv, pred_cls))
            else:
                metrics["pr_auc"] = None
                metrics["accuracy_at_threshold_0_5"] = None
                metrics["label_note"] = "Недостаточно строк с label 0/1 или только один класс"
        else:
            metrics["pr_auc"] = None
            metrics["accuracy_at_threshold_0_5"] = None
            metrics["label_note"] = "Колонка label не найдена — PR-AUC не считается"

        preview_cols = [c for c in (prompt_col, resp_col, "hallucination_prob") if c in out_df.columns]
        preview = out_df[preview_cols].head(100).to_dict(orient="records")

        return {
            "ok": True,
            "filename": file.filename,
            "metrics": metrics,
            "preview": preview,
            "preview_note": f"Показаны первые {min(100, len(out_df))} из {len(out_df)} строк",
        }

    return router
