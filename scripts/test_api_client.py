#!/usr/bin/env python3
"""Проверка POST /predict без дополнительных зависимостей (urllib)."""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    base = os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")
    payload = {
        "prompt": os.environ.get("TEST_PROMPT", "Сколько будет 2+2?"),
        "response": os.environ.get("TEST_RESPONSE", "5"),
        "id": "test_client",
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/predict",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            print(resp.status, resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(e.code, e.read().decode("utf-8"), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
