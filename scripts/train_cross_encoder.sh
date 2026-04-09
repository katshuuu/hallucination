#!/usr/bin/env bash
# Опционально: старый CrossEncoder (в configs/default.yaml выставьте backend: cross_encoder)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
exec python -m hallucination_detector.train --config "$ROOT/configs/default.yaml"
