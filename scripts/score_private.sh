#!/usr/bin/env bash
# Скоринг приватного бенча: knowledge_bench_private.csv -> knowledge_bench_private_scores.csv
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IN="${ROOT}/data/bench/knowledge_bench_private.csv"
OUT="${ROOT}/data/bench/knowledge_bench_private_scores.csv"

if [[ ! -f "$IN" ]]; then
  echo "Missing input: $IN" >&2
  exit 1
fi

# shellcheck source=/dev/null
source .venv/bin/activate
python -m hallucination_detector.score_csv \
  --config "$ROOT/configs/default.yaml" \
  --input "$IN" \
  --output "$OUT"

echo "Wrote: $OUT"
