#!/usr/bin/env python3
"""
Опциональная слабая разметка через LLM (только при сборе данных, не в инференсе).

Требует: pip install openai (опциональная зависимость) и переменную окружения OPENAI_API_KEY.

Выход: CSV с исходными prompt/response + judge_label + judge_rationale + сохранённый JSONL с сырыми ответами API.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("data/judge_runs"))
    p.add_argument("--model", type=str, default="gpt-4o-mini")
    p.add_argument("--max-rows", type=int, default=200)
    args = p.parse_args()

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY не задан — пропуск judge. Это нормально для финальной автономной поставки.")
        return

    try:
        from openai import OpenAI
    except ImportError:
        print("Установите openai: pip install openai")
        return

    client = OpenAI(api_key=key)
    df = pd.read_csv(args.input).head(args.max_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = args.output_dir / f"judge_raw_{stamp}.jsonl"
    rows_out = []

    with raw_path.open("w", encoding="utf-8") as raw_f:
        for i, row in df.iterrows():
            prompt = str(row.get("prompt", ""))
            response = str(row.get("response", row.get("model_answer", row.get("answer", ""))))
            user_msg = (
                "Оцени, содержит ли ответ фактическую галлюцинацию относительно вопроса. "
                "Ответь строго JSON: {\"hallucination\": true/false, \"rationale\": \"...\"}\n\n"
                f"Вопрос: {prompt}\nОтвет: {response}"
            )
            comp = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "user", "content": user_msg}],
                temperature=0,
            )
            text = comp.choices[0].message.content or ""
            raw_f.write(
                json.dumps({"i": int(i), "prompt": prompt, "response": response, "raw": text}, ensure_ascii=False)
                + "\n"
            )
            try:
                parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
                j_lab = 1 if parsed.get("hallucination") else 0
                rat = str(parsed.get("rationale", ""))
            except Exception:
                j_lab, rat = 0, "parse_error"

            rows_out.append(
                {
                    **row.to_dict(),
                    "judge_label": j_lab,
                    "judge_rationale": rat,
                    "judge_model": args.model,
                }
            )

    out_csv = args.output_dir / f"with_judge_{stamp}.csv"
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} and {raw_path}")


if __name__ == "__main__":
    main()
