"""
FastAPI-приложение: модель загружается один раз при старте (lifespan).
Документация: GET /docs (Swagger).

Промышленные настройки: CORS (HALLUCINATION_CORS_ORIGINS), X-Request-ID,
раздельные liveness/readiness, версия API.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from hallucination_detector.api.schemas import PredictRequest, PredictResponse
from hallucination_detector.api.web_ui import STATIC_DIR, create_ui_router
from hallucination_detector.detection_timing import build_timing_result, exceeds_jury_limit
from hallucination_detector.settings import load_config, repo_root
from hallucination_detector.tabular.scorer import TabularScorer, load_scorer_from_config

logger = logging.getLogger("hallucination_detector.api")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

_scorer: TabularScorer | None = None
_cols: dict[str, Any] | None = None
_run_cfg: dict[str, Any] | None = None


def _app_version() -> str:
    try:
        return pkg_version("hallucination-detector")
    except PackageNotFoundError:
        return "0.0.0-dev"


def _config_path() -> Path:
    env = os.environ.get("HALLUCINATION_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "configs" / "default.yaml"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scorer, _cols, _run_cfg
    cfg_path = _config_path()
    if not cfg_path.is_file():
        logger.error("Конфиг не найден: %s", cfg_path)
        raise RuntimeError(f"Config not found: {cfg_path}")
    cfg = load_config(cfg_path)
    if cfg.get("backend") != "tabular":
        raise RuntimeError("API поддерживает только backend: tabular (см. configs/default.yaml)")
    try:
        _scorer = load_scorer_from_config(cfg)
        _cols = cfg["columns"]
        _run_cfg = cfg
        logger.info("Модель загружена из %s", cfg["paths"]["artifact_dir"])
    except FileNotFoundError as e:
        logger.error("%s", e)
        raise RuntimeError(
            "Артефакт tabular_bundle.joblib не найден. Выполните ./scripts/train.sh"
        ) from e
    yield
    _scorer = None
    _cols = None
    _run_cfg = None
    logger.info("Остановка приложения")


app = FastAPI(
    title="Hallucination detector API",
    description=(
        "Скоринг пары prompt/response через табличный стек (LightGBM + признаки). "
        "Предназначено для офлайн-проверки ответов LLM в корпоративном контуре (без внешних LLM API на инференсе)."
    ),
    version=_app_version(),
    lifespan=lifespan,
)

_origins = [x.strip() for x in os.environ.get("HALLUCINATION_CORS_ORIGINS", "").split(",") if x.strip()]
if _origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
app.add_middleware(RequestIdMiddleware)


def _tabular_state() -> tuple[TabularScorer | None, dict[str, Any] | None, dict[str, Any] | None]:
    return _scorer, _cols, _run_cfg


app.mount("/ui/static", StaticFiles(directory=str(STATIC_DIR)), name="ui_static")
app.include_router(create_ui_router(_tabular_state))


@app.get("/health")
@app.get("/health/live")
def health_live() -> dict[str, str]:
    """Liveness: процесс жив (для Kubernetes/docker healthcheck)."""
    return {"status": "ok"}


@app.get("/health/ready")
def health_ready() -> dict[str, str]:
    """Readiness: модель загружена и готова к /predict."""
    if _scorer is None or _cols is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ready", "model": "loaded"}


@app.get("/version")
def version_info() -> dict[str, str]:
    """Версия сервиса и путь к конфигу (без секретов)."""
    return {
        "service": "hallucination-detector",
        "version": _app_version(),
        "config": str(_config_path()),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """
    Возвращает `hallucination_prob` — вероятность положительного класса (галлюцинация / явная ошибка).
    """
    if _scorer is None or _cols is None or _run_cfg is None:
        raise HTTPException(status_code=503, detail="Модель не инициализирована")

    pc = _cols["prompt"]
    rc = _cols["response"]
    t0 = time.perf_counter()
    try:
        score = _scorer.score_one(body.prompt, body.response, pc, rc)
    except Exception as e:
        rid = getattr(request.state, "request_id", None)
        logger.exception("Ошибка предсказания id=%s request_id=%s", body.id, rid)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}") from e

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    timing = build_timing_result(elapsed_ms, _run_cfg)
    rid = getattr(request.state, "request_id", None)
    if exceeds_jury_limit(timing):
        logger.warning(
            "predict billable_ms=%.2f > jury_max=%.2f id=%s request_id=%s",
            timing.billable_time_ms,
            timing.jury_max_billable_ms,
            body.id,
            rid,
        )
    logger.info(
        "predict id=%s request_id=%s prob=%.6f wall_ms=%.2f billable_ms=%.2f prompt_len=%d response_len=%d",
        body.id,
        rid,
        score,
        timing.wall_time_ms,
        timing.billable_time_ms,
        len(body.prompt),
        len(body.response),
    )
    return PredictResponse(
        hallucination_prob=float(score),
        id=body.id,
        wall_time_ms=round(timing.wall_time_ms, 3),
        billable_time_ms=round(timing.billable_time_ms, 3),
        jury_max_billable_ms=round(timing.jury_max_billable_ms, 3),
        inference_time_ms=round(timing.wall_time_ms, 3),
    )
