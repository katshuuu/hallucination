#!/usr/bin/env bash
# Обучение табличного стека (LightGBM + признаки); артефакты → model/tabular/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "${ROOT}/data/processed/train_merged.csv" ]]; then
  echo "Нет data/processed/train_merged.csv — сначала ./scripts/build_train_data.sh" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
exec python -m hallucination_detector.tabular.train --config "$ROOT/configs/default.yaml"
