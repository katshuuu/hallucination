#!/usr/bin/env bash
# Сборка data/processed/train_merged.csv из открытых источников + синтетики + manual (см. configs/data_pipeline.yaml).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
exec python -m hallucination_detector.data.build --config "$ROOT/configs/data_pipeline.yaml" "$@"
