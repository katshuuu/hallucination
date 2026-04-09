#!/usr/bin/env python3
"""
Проверка соответствия ограничениям соревнования (без внешних API на инференсе, конфиг).
Запуск: python scripts/check_constraints.py [--config configs/default.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from hallucination_detector.settings import load_config, repo_root


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=repo_root() / "configs" / "default.yaml")
    args = p.parse_args()
    cfg = load_config(args.config)

    print("=== Ограничения и конфигурация ===\n")
    print(f"backend: {cfg.get('backend', 'tabular')}")
    print(f"artifact_dir: {cfg['paths'].get('artifact_dir')}")
    print(f"embedding_model: {cfg.get('tabular', {}).get('embedding_model')}")
    print(f"use_lm_features (локальный LM при инференсе): {cfg.get('tabular', {}).get('use_lm_features', False)}")
    print(f"tfidf_max_features: {cfg.get('tabular', {}).get('tfidf_max_features')}")

    # Инференс не должен требовать ключей к облачным API
    leak_keys = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
    found = [k for k in leak_keys if os.environ.get(k)]
    if found:
        print(f"\n[warn] Заданы переменные окружения (для инференса не нужны): {found}")
    else:
        print("\n[ok] Ключи облачных LLM в окружении не обнаружены (для скоринга не требуются).")

    bundle = Path(cfg["paths"]["artifact_dir"]) / "tabular_bundle.joblib"
    if bundle.is_file():
        print(f"\n[ok] Артефакт найден: {bundle}")
    else:
        print(f"\n[!!] Нет {bundle} — выполните ./scripts/train.sh перед замером скорости/скоринга.")

    try:
        import torch

        print(f"\nCUDA доступна: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"\n[torch] {e}")

    tm = cfg.get("timing") or {}
    print("\n=== Время детекции (жюри) ===")
    print(f"jury_max_billable_ms: {tm.get('jury_max_billable_ms', 500)}")
    print(f"reference_hardware: {tm.get('reference_hardware', 'см. docs/COMPETITION_RULES.md')}")
    print(f"gigachat_jury_model_id: {tm.get('gigachat_jury_model_id', 'ai-sage/GigaChat3-10B-A1.8B-bf16')}")
    print(f"uses_gradient_analysis: {tm.get('uses_gradient_analysis', False)}")

    print("\n=== Рекомендация ===")
    print("Скоринг и API работают только с локальными весами и HF-моделями из кэша.")
    print("Внешние HTTP API для «проверки фактов» на инференсе не используются.")
    print("Полные правила лимита 500 ms и billable time: docs/COMPETITION_RULES.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
