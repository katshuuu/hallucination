"""
Правила замера времени детекции галлюцинаций (жюри / production).

Ориентиры из регламента:
- бюджет: не более 500 ms на один ответ (по «учитываемому» времени, см. ниже);
- финальная проверка: 1× GPU A100 80GB;
- метрика качества: PR-AUC; при равенстве — скорость алгоритма;
- один forward модели ``ai-sage/GigaChat3-10B-A1.8B-bf16`` на последовательности prompt+response
  может не входить в учитываемое время только для снятия активаций/проб (не для методов с градиентами);
- backward pass при анализе градиентов учитывается всегда, независимо от модели.

Текущий табличный стек (MiniLM + LightGBM) не использует GigaChat: всё измеренное время — учитываемое.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Модель, для которой жюри разрешает исключить один forward из «учитываемого» времени (см. регламент).
GIGACHAT_JURY_MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"

# Целевой верхний предел учитываемого времени на одну пару (мс).
JURY_MAX_BILLABLE_MS = 500.0

# Эталонное железо для финальной оценки (документация).
JURY_REFERENCE_GPU = "1× NVIDIA A100 80GB"


@dataclass(frozen=True)
class DetectionTimingResult:
    """Результат одного вызова детектора."""

    wall_time_ms: float
    """Полное время от входа в скорер до получения вероятности (wall clock)."""

    billable_time_ms: float
    """Время, которое сравнивают с лимитом 500 ms по правилам жюри."""

    jury_max_billable_ms: float
    uses_gradient_analysis: bool
    gigachat_forward_excluded_ms: float


def timing_config_defaults() -> dict[str, Any]:
    return {
        "jury_max_billable_ms": JURY_MAX_BILLABLE_MS,
        "reference_hardware": JURY_REFERENCE_GPU,
        "gigachat_jury_model_id": GIGACHAT_JURY_MODEL_ID,
        # Сколько миллисекунд одного forward GigaChat на prompt+response вычитается из billable (0 = не вычитать).
        "gigachat_forward_excluded_ms": 0.0,
        # True, если пайплайн использует backward/градиенты — тогда вычитание forward не применяется, всё время учитывается.
        "uses_gradient_analysis": False,
    }


def merge_timing_config(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    out = timing_config_defaults()
    if cfg is None:
        return out
    raw = cfg.get("timing")
    if not isinstance(raw, Mapping):
        return out
    for k, v in raw.items():
        if k in out:
            out[k] = v
    return out


def compute_billable_ms(
    wall_time_ms: float,
    *,
    uses_gradient_analysis: bool,
    gigachat_forward_excluded_ms: float,
) -> float:
    """
    Учитываемое время для сравнения с 500 ms.

    Если есть анализ градиентов — учитывается всё wall time (включая backward).
    Иначе из wall time вычитается не более одного заранее заданного forward GigaChat (мс), если настроено.
    """
    if uses_gradient_analysis:
        return float(wall_time_ms)
    ex = max(0.0, float(gigachat_forward_excluded_ms))
    return max(0.0, float(wall_time_ms) - ex)


def build_timing_result(
    wall_time_ms: float,
    cfg: Mapping[str, Any] | None,
) -> DetectionTimingResult:
    t = merge_timing_config(cfg)
    uses = bool(t.get("uses_gradient_analysis", False))
    excluded = float(t.get("gigachat_forward_excluded_ms", 0.0))
    billable = compute_billable_ms(
        wall_time_ms,
        uses_gradient_analysis=uses,
        gigachat_forward_excluded_ms=excluded,
    )
    return DetectionTimingResult(
        wall_time_ms=float(wall_time_ms),
        billable_time_ms=billable,
        jury_max_billable_ms=float(t.get("jury_max_billable_ms", JURY_MAX_BILLABLE_MS)),
        uses_gradient_analysis=uses,
        gigachat_forward_excluded_ms=excluded,
    )


def exceeds_jury_limit(result: DetectionTimingResult) -> bool:
    return result.billable_time_ms > result.jury_max_billable_ms
