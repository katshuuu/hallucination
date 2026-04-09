from __future__ import annotations

import argparse
import random
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from hallucination_detector.data.dataset_contract import validate_train_csv
from hallucination_detector.data.normalize import clean_text
from hallucination_detector.data.source_tags import MANUAL, SUPPLEMENTAL, normalize_source
from hallucination_detector.data.synthetic import generate_synthetic_rows
from hallucination_detector.settings import repo_root


def _dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (r["prompt"], r["response"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def build_train_table(cfg: dict[str, Any], root: Path) -> pd.DataFrame:
    seed = int(cfg.get("seed", 42))
    rng = random.Random(seed)
    limits = cfg.get("limits", {})
    offline = bool(cfg.get("offline_only", False))

    rows: list[dict[str, Any]] = []
    hf_ok = False

    if not offline:
        try:
            from hallucination_detector.data.hf_sources import load_paws, load_snli, load_truthful_qa

            rows.extend(load_truthful_qa(int(limits.get("truthful_qa") or 12000)))
            rows.extend(load_snli(int(limits.get("snli") or 8000)))
            rows.extend(load_paws(int(limits.get("paws") or 6000)))
            hf_ok = True
        except Exception as e:
            print(f"[warn] HF sources failed ({e}); using offline-only bundle.")

    syn_n = int(limits.get("synthetic") or 2500)
    if not hf_ok:
        syn_n = max(syn_n, 3000)
    rows.extend(generate_synthetic_rows(syn_n, rng))

    manual_path = root / "data" / "validation" / "manual_labeled_sample.csv"
    if manual_path.is_file():
        mdf = pd.read_csv(manual_path)
        for _, r in mdf.iterrows():
            row = {
                "prompt": clean_text(r["prompt"]),
                "response": clean_text(r["response"]),
                "label": int(r["label"]),
                "source": MANUAL,
            }
            if "id" in mdf.columns and pd.notna(r.get("id")):
                row["id"] = str(r["id"]).strip()
            rows.append(row)

    supplemental_path = root / "data" / "supplemental" / "labeled_qa_diverse_v1.csv"
    if supplemental_path.is_file():
        sdf = pd.read_csv(supplemental_path)
        for _, r in sdf.iterrows():
            row = {
                "prompt": clean_text(r["prompt"]),
                "response": clean_text(r["response"]),
                "label": int(r["label"]),
                "source": SUPPLEMENTAL,
            }
            if "id" in sdf.columns and pd.notna(r.get("id")):
                row["id"] = str(r["id"]).strip()
            rows.append(row)

    rows = _dedupe(rows)
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int).clip(0, 1)
    df["source"] = df["source"].map(normalize_source)

    if "id" not in df.columns:
        df.insert(0, "id", [str(uuid.uuid4()) for _ in range(len(df))])
    else:
        missing = df["id"].isna() | (df["id"].astype(str).str.strip() == "")
        if missing.any():
            df.loc[missing, "id"] = [str(uuid.uuid4()) for _ in range(int(missing.sum()))]
        df["id"] = df["id"].astype(str)

    df = df[["id", "prompt", "response", "label", "source"]]
    return df


def main() -> None:
    p = argparse.ArgumentParser(description="Build prompt/response/label train CSV from multiple sources")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "data_pipeline.yaml",
    )
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--offline",
        action="store_true",
        help="Не загружать Hugging Face: только синтетика, manual и supplemental (воспроизводимо без сети).",
    )
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.offline:
        cfg["offline_only"] = True
    root = repo_root()
    out = Path(args.output or cfg["output_csv"])
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_train_table(cfg, root)
    df.to_csv(out, index=False)
    validate_train_csv(out)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["source"].value_counts().to_string())


if __name__ == "__main__":
    main()
