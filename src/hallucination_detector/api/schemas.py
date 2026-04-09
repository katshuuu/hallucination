"""Pydantic-схемы входа/выхода для /predict (валидация без «сырых» dict)."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Одна пара вопрос–ответ; модель сама строит признаки из текстов."""

    prompt: str = Field(..., min_length=1, max_length=100_000, description="Вопрос или контекст")
    response: str = Field(..., min_length=1, max_length=100_000, description="Ответ для проверки")
    id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Необязательный идентификатор запроса со стороны клиента",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "Столица Франции?",
                    "response": "Париж",
                    "id": "req-001",
                }
            ]
        }
    }


class PredictResponse(BaseModel):
    """Вероятность галлюцинации (положительный класс)."""

    hallucination_prob: float = Field(..., ge=0.0, le=1.0)
    id: Optional[str] = Field(default=None, description="Тот же id, что в запросе, если был передан")
    wall_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Полное время обработки запроса детектором (wall clock), мс",
    )
    billable_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Учитываемое время по правилам жюри (сравнение с лимитом), мс",
    )
    jury_max_billable_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Лимит учитываемого времени из конфигурации (обычно 500 мс)",
    )
    inference_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Устаревший алиас: равно wall_time_ms (обратная совместимость)",
    )


class ErrorDetail(BaseModel):
    detail: str
