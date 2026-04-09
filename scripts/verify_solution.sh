#!/usr/bin/env bash
# Смоук-тест цепочки для самопроверки перед сдачей (аудит данных → скоринг → замер → ограничения).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Сначала: ./scripts/install.sh" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

echo "========== 1. Контракт train CSV =========="
python scripts/audit_train_csv.py
python scripts/check_no_public_benchmark_overlap.py

echo ""
echo "========== 2. Ограничения (API, конфиг) =========="
python scripts/check_constraints.py

echo ""
echo "========== 3. Модель и скоринг =========="
BUNDLE="${ROOT}/model/tabular/tabular_bundle.joblib"
if [[ -f "$BUNDLE" ]]; then
  OUT="${TMPDIR:-/tmp}/jury_smoke_scores.csv"
  python -m hallucination_detector.score_csv \
    --config "${ROOT}/configs/default.yaml" \
    --input "${ROOT}/data/bench/knowledge_bench_private.csv" \
    --output "$OUT"
  echo "Скоринг OK → $OUT"
  head -5 "$OUT"

  echo ""
  echo "========== 4. Латентность (cli benchmark) =========="
  python -m hallucination_detector.cli benchmark --repeats 30

  echo ""
  echo "========== 5. Детальный замер (measure_performance) =========="
  python scripts/measure_performance.py --repeats 40 --csv "${ROOT}/data/bench/knowledge_bench_private.csv"
else
  echo "Пропуск скоринга/бенчмарка: нет $BUNDLE — выполните ./scripts/train.sh"
fi

echo ""
echo "Готово. Подробности: docs/TESTING_JURY.md"
