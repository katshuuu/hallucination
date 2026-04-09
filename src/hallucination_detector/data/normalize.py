from __future__ import annotations


def clean_text(s: str, *, max_chars: int = 4000) -> str:
    s = " ".join(str(s).split())
    if len(s) > max_chars:
        s = s[:max_chars]
    return s
