#!/usr/bin/env python3
"""
Замер латентности и памяти процесса при инференсе (табличный скорер).

Примеры:
  python scripts/measure_performance.py --repeats 50
  python scripts/measure_performance.py --csv data/bench/knowledge_bench_private.csv

Память (RSS): на Linux ru_maxrss в KiB, на macOS в байтах — учтено.
Опционально точнее: pip install psutil
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hallucination_detector.detection_timing import JURY_MAX_BILLABLE_MS, JURY_REFERENCE_GPU, build_timing_result
from hallucination_detector.settings import load_config, repo_root
from hallucination_detector.tabular.scorer import load_scorer_from_config


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except ImportError:
        u = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            return float(u) / (1024**2)
        return float(u) / 1024.0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs" / "default.yaml")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--csv", type=Path, default=None, help="Прогнать батч-скоринг всего CSV")
    args = p.parse_args()

    cfg = load_config(args.config)
    if cfg.get("backend") != "tabular":
        print("Скрипт заточен под backend: tabular", file=sys.stderr)
        return 1

    bundle = Path(cfg["paths"]["artifact_dir"]) / "tabular_bundle.joblib"
    if not bundle.is_file():
        print("Нет модели:", bundle, file=sys.stderr)
        return 1

    rss0 = _rss_mb()
    scorer = load_scorer_from_config(cfg)
    cols = cfg["columns"]
    pc, rc = cols["prompt"], cols["response"]
    rss1 = _rss_mb()

    prompt = "Какая столица у Франции?"
    response = "Париж."

    for _ in range(args.warmup):
        _ = scorer.score_one(prompt, response, pc, rc)

    try:
        import torch

        _torch = torch
        sync = torch.cuda.is_available()
    except Exception:
        _torch = None
        sync = False

    times: list[float] = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        _ = scorer.score_one(prompt, response, pc, rc)
        if sync and _torch is not None:
            _torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(times, dtype=np.float64)
    rss2 = _rss_mb()

    print("=== Одиночный скоринг (одна пара) ===")
    print(f"mean_ms={arr.mean():.2f}  p95_ms={np.percentile(arr, 95):.2f}  std_ms={arr.std():.2f}")
    print(f"repeats={args.repeats} warmup={args.warmup}")
    bill = np.array([build_timing_result(float(w), cfg).billable_time_ms for w in times], dtype=np.float64)
    print(
        f"billable_ms (jury): mean={bill.mean():.2f}  p95={np.percentile(bill, 95):.2f}  "
        f"(лимит {JURY_MAX_BILLABLE_MS:.0f} ms; эталон {JURY_REFERENCE_GPU}; см. docs/COMPETITION_RULES.md)"
    )
    print(f"RSS МиБ: до загрузки модели≈{rss0:.1f}, после загрузки≈{rss1:.1f}, после прогрева≈{rss2:.1f}")

    if args.csv and args.csv.is_file():
        df = pd.read_csv(args.csv)
        n = len(df)
        t0 = time.perf_counter()
        from hallucination_detector.columns_util import response_column

        resp_col = response_column(df, cols)
        probs = scorer.predict_proba_positive(df, pc, resp_col)
        elapsed = time.perf_counter() - t0
        print(f"\n=== Батч: {args.csv} ({n} строк) ===")
        print(f"total_s={elapsed:.3f}  ms_per_row={elapsed / max(n, 1) * 1000:.2f}")
        _ = probs  # use
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
