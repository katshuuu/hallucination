#!/usr/bin/env python3
"""
Проверка train CSV на контракт «тексты + метка + source» (без анонимных только-фич).
Использование: python scripts/audit_train_csv.py [path/to/train_merged.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

from hallucination_detector.data.dataset_contract import validate_train_csv
from hallucination_detector.settings import repo_root


def main() -> None:
    default = repo_root() / "data" / "processed" / "train_merged.csv"
    path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else default
    validate_train_csv(path)
    print("OK:", path)


if __name__ == "__main__":
    main()
