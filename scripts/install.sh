#!/usr/bin/env bash
# Установка зависимостей и подготовка окружения (загрузка базовой модели в HF cache при первом запуске).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

if [[ "${SKIP_PRELOAD:-}" != "1" ]]; then
  echo "Preloading CrossEncoder weights (cached under ~/.cache/huggingface)..."
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
  python -c "from sentence_transformers.cross_encoder import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
fi

echo "Done. Activate venv: source .venv/bin/activate"
