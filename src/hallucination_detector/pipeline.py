from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import CrossEncoder


class HallucinationDetector:
    """
    Быстрый скоринг пар (prompt, response) через CrossEncoder без внешних API.

    Для num_labels=1 sentence-transformers применяет Sigmoid к логиту — выход predict уже в [0,1].
    Для num_labels=2 используем softmax и столбец positive_class_index.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_length: int = 256,
        device: str | None = None,
        batch_size: int = 64,
        amp: bool = True,
        num_labels: int = 1,
        positive_class_index: int = 1,
    ) -> None:
        self.batch_size = batch_size
        self.amp = amp and torch.cuda.is_available()
        self.num_labels = num_labels
        self.positive_class_index = positive_class_index
        self._model = CrossEncoder(
            model_name_or_path,
            max_length=max_length,
            num_labels=num_labels,
            device=device,
        )

    @property
    def device(self) -> torch.device:
        return self._model.device

    def predict_logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Сырые логиты (без активации выходного слоя)."""
        texts = [[p, a] for p, a in pairs]
        return np.asarray(
            self._model.predict(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                activation_fn=nn.Identity(),
            ),
            dtype=np.float64,
        ).reshape(len(texts), -1)

    def predict_proba(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """P(положительный класс): для num_labels=1 — уже сигмоида внутри CrossEncoder.predict."""
        texts = [[p, a] for p, a in pairs]
        if self.num_labels == 1:
            out = self._model.predict(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            arr = np.asarray(out, dtype=np.float64).reshape(-1)
            return arr
        out = self._model.predict(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )
        arr = np.asarray(out, dtype=np.float64)
        if arr.ndim == 1:
            return arr
        return arr[:, self.positive_class_index]

    def score_one(self, prompt: str, response: str) -> float:
        """Одна пара — float в [0,1]."""
        return float(self.predict_proba([(prompt, response)])[0])

    def benchmark_latency(
        self,
        prompt: str,
        response: str,
        *,
        warmup: int = 5,
        repeats: int = 50,
    ) -> dict[str, float]:
        """Среднее и p95 времени одного ответа (синхронизация CUDA)."""
        for _ in range(warmup):
            _ = self.score_one(prompt, response)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = self.score_one(prompt, response)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        return {
            "mean_ms": float(arr.mean()),
            "p95_ms": float(np.percentile(arr, 95)),
            "std_ms": float(arr.std()),
        }


def load_detector_from_config(cfg: dict) -> HallucinationDetector:
    m = cfg["model"]
    inf = cfg["inference"]
    device = inf.get("device") or None
    raw_name = m["name"]
    p = Path(raw_name).expanduser()
    if p.exists():
        path = str(p.resolve())
    elif cfg.get("training", {}).get("base_model"):
        base = cfg["training"]["base_model"]
        warnings.warn(
            f"Артефакт модели не найден ({raw_name}), используется базовая модель {base!r} без дообучения.",
            stacklevel=2,
        )
        path = base
    else:
        path = raw_name
    return HallucinationDetector(
        path,
        max_length=int(m.get("max_length", 256)),
        device=device,
        batch_size=int(inf.get("batch_size", 64)),
        amp=bool(inf.get("amp", True)),
        num_labels=int(m.get("num_labels", 1)),
        positive_class_index=int(m.get("positive_class_index", 1)),
    )
