from __future__ import annotations

import argparse
from pathlib import Path

from hallucination_detector.benchmark import run_benchmark
from hallucination_detector.score_csv import run_score
from hallucination_detector.settings import repo_root
from hallucination_detector.tabular.train import run_train_tabular


def main() -> None:
    default_cfg = repo_root() / "configs" / "default.yaml"
    parser = argparse.ArgumentParser(prog="hallucination-detector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train tabular LightGBM stack")
    p_train.add_argument("--config", type=Path, default=default_cfg)

    p_score = sub.add_parser("score", help="Score a CSV file")
    p_score.add_argument("--config", type=Path, default=default_cfg)
    p_score.add_argument("--input", type=Path, required=True)
    p_score.add_argument("--output", type=Path, required=True)
    p_score.add_argument("--model", type=str, default=None)

    p_bench = sub.add_parser("benchmark", help="Latency benchmark")
    p_bench.add_argument("--config", type=Path, default=default_cfg)
    p_bench.add_argument("--prompt", type=str, default="What is the capital of France?")
    p_bench.add_argument("--response", type=str, default="Paris is the capital.")
    p_bench.add_argument("--answer", type=str, default=None)
    p_bench.add_argument("--warmup", type=int, default=5)
    p_bench.add_argument("--repeats", type=int, default=50)

    args = parser.parse_args()
    if args.cmd == "train":
        run_train_tabular(args.config)
    elif args.cmd == "score":
        run_score(args.config, args.input, args.output, model_override=args.model)
    elif args.cmd == "benchmark":
        resp = args.response if getattr(args, "answer", None) is None else args.answer
        run_benchmark(
            args.config,
            prompt=args.prompt,
            response=resp,
            warmup=args.warmup,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
