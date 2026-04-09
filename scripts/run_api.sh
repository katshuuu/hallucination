#!/usr/bin/env bash
# Запуск HTTP API: http://127.0.0.1:8000/docs
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
export HALLUCINATION_CONFIG="${HALLUCINATION_CONFIG:-$ROOT/configs/default.yaml}"
exec uvicorn main:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" "$@"
