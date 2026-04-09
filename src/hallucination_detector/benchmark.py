from __future__ import annotations

import argparse
from pathlib import Path

from hallucination_detector.detection_timing import build_timing_result
from hallucination_detector.settings import load_config, repo_root


def run_benchmark(
    config_path: Path,
    *,
    prompt: str,
    response: str,
    warmup: int = 5,
    repeats: int = 50,
) -> None:
    cfg = load_config(config_path)
    backend = cfg.get("backend", "tabular")

    if backend == "tabular":
        from hallucination_detector.tabular.scorer import load_scorer_from_config

        scorer = load_scorer_from_config(cfg)
        cols = cfg["columns"]
        pc, rc = cols["prompt"], cols["response"]

        for _ in range(warmup):
            _ = scorer.score_one(prompt, response, pc, rc)
        import time

        import numpy as np
        import torch

        times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = scorer.score_one(prompt, response, pc, rc)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        arr = np.array(times, dtype=np.float64)
        billables = np.array([build_timing_result(w, cfg).billable_time_ms for w in times], dtype=np.float64)
        tmax = float(cfg.get("timing", {}).get("jury_max_billable_ms", 500))
        print(
            f"Latency (tabular): mean={arr.mean():.2f} ms, p95={np.percentile(arr, 95):.2f} ms (wall)"
        )
        print(
            f"Billable (jury): mean={billables.mean():.2f} ms, p95={np.percentile(billables, 95):.2f} ms "
            f"(limit {tmax:.0f} ms; финальная оценка на A100 — см. docs/COMPETITION_RULES.md)"
        )
        if np.percentile(billables, 95) > tmax:
            print(
                f"[warn] p95 billable {np.percentile(billables, 95):.2f} ms > {tmax:.0f} ms — "
                "проверьте на эталонном GPU; при методах с GigaChat настройте timing в configs/default.yaml"
            )
        print(f"Example score: {scorer.score_one(prompt, response, pc, rc):.6f}")
        return

    from hallucination_detector.pipeline import load_detector_from_config

    det = load_detector_from_config(cfg)
    stats = det.benchmark_latency(prompt, response, warmup=warmup, repeats=repeats)
    print(f"Latency (cross_encoder): mean={stats['mean_ms']:.2f} ms, p95={stats['p95_ms']:.2f} ms")
    print(f"Example score: {det.score_one(prompt, response):.6f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Measure latency of one score() call")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
    )
    p.add_argument("--prompt", type=str, default="What is the capital of France?")
    p.add_argument("--response", type=str, default="Paris is the capital of France.")
    p.add_argument("--answer", type=str, default=None, help="Alias for --response")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    resp = args.response if args.answer is None else args.answer
    run_benchmark(
        args.config,
        prompt=args.prompt,
        response=resp,
        warmup=args.warmup,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    main()
