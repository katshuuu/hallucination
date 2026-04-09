#!/usr/bin/env python3
"""
Демонстрация: признаки восстанавливаются из текстов prompt/response тем же экстрактором,
что в bundle после train (без «магических» заранее посчитанных столбцов в train CSV).

Требуется обученный model/tabular/tabular_bundle.joblib (./scripts/train.sh).

Пример:
  python scripts/reproduce_features_from_csv.py --print-hash --row-index 0
  python scripts/reproduce_features_from_csv.py --row-id m001
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import joblib
import pandas as pd

from hallucination_detector.settings import load_config, repo_root


def main() -> None:
    root = repo_root()
    p = argparse.ArgumentParser(description="Показать первые признаки из текстовой строки train CSV")
    p.add_argument("--config", type=Path, default=root / "configs" / "default.yaml")
    p.add_argument("--csv", type=Path, default=root / "data" / "processed" / "train_merged.csv")
    p.add_argument("--bundle-dir", type=Path, default=None, help="Каталог с tabular_bundle.joblib")
    p.add_argument("--row-id", type=str, default=None)
    p.add_argument("--row-index", type=int, default=0)
    p.add_argument("--print-hash", action="store_true", help="SHA-256 от нормализованной пары текстов")
    p.add_argument("--top", type=int, default=16, help="Сколько первых признаков вывести")
    args = p.parse_args()

    cfg = load_config(args.config)
    bdir = Path(args.bundle_dir or cfg["paths"]["artifact_dir"])
    bundle_path = bdir / "tabular_bundle.joblib"
    if not bundle_path.is_file():
        print("Нет", bundle_path, "— выполните ./scripts/train.sh", file=sys.stderr)
        sys.exit(1)

    bundle = joblib.load(bundle_path)
    ext = bundle["extractor"]
    names = list(bundle.get("feature_names") or ext.feature_names)

    df = pd.read_csv(args.csv)
    if args.row_id is not None:
        sub = df[df["id"].astype(str) == args.row_id]
        if sub.empty:
            print("id не найден:", args.row_id, file=sys.stderr)
            sys.exit(1)
    else:
        sub = df.iloc[[args.row_index]]

    pr = str(sub.iloc[0]["prompt"])
    rs = str(sub.iloc[0]["response"])
    if args.print_hash:
        h = hashlib.sha256(f"{pr}\n||\n{rs}".encode("utf-8")).hexdigest()
        print("sha256(prompt||response):", h)

    X = ext.transform(sub, "prompt", "response")
    dense = X.toarray().ravel()
    n = min(args.top, len(dense))
    print(f"Вектор признаков: dim={len(dense)} (первые {n} имён и значений)")
    for i in range(n):
        label = names[i] if i < len(names) else f"f_{i}"
        print(f"  {label}: {dense[i]:.6g}")


if __name__ == "__main__":
    main()
