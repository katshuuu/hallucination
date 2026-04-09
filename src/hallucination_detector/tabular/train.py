from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

from hallucination_detector.columns_util import response_column
from hallucination_detector.data.dataset_contract import count_pair_overlap, validate_train_csv
from hallucination_detector.settings import load_config, repo_root
from hallucination_detector.tabular.extractor import TabularFeatureExtractor


def run_train_tabular(config_path: Path) -> None:
    cfg = load_config(config_path)
    if cfg.get("backend") != "tabular":
        raise ValueError("В configs/default.yaml укажите backend: tabular")

    tab = cfg["tabular"]
    paths = cfg["paths"]
    cols = cfg["columns"]
    tr = cfg.get("training", {})

    train_csv = Path(paths["train_csv"])
    if not train_csv.is_file():
        raise FileNotFoundError(train_csv)

    validate_train_csv(train_csv)
    public_bench = repo_root() / "knowledge_bench_public.csv"
    if public_bench.is_file():
        overlap = count_pair_overlap(
            train_csv,
            public_bench,
            other_prompt_col="prompt",
            other_response_col="model_answer",
        )
        if overlap > 0 and not bool(tr.get("allow_public_overlap", False)):
            raise ValueError(
                f"Обнаружено пересечение train с public benchmark: {overlap} пар. "
                "Удалите эти пары из train или явно разрешите training.allow_public_overlap=true "
                "(не рекомендуется для финальной сдачи)."
            )

    df = pd.read_csv(train_csv)
    prompt_col = cols["prompt"]
    resp_col = response_column(df, cols)
    label_col = cols["label"]

    jc_default = (
        "judge_relevance",
        "judge_factuality",
        "judge_consistency",
        "judge_completeness",
    )
    jc = tab.get("judge_columns") or list(jc_default)
    ext = TabularFeatureExtractor(
        embedding_model=tab.get("embedding_model", "all-MiniLM-L6-v2"),
        tfidf_max_features=int(tab.get("tfidf_max_features", 1500)),
        use_lm_features=bool(tab.get("use_lm_features", False)),
        lm_model_name=tab.get("lm_model_name", "distilgpt2"),
        judge_columns=tuple(jc),
    )
    y = df[label_col].astype(int).values
    test_size = float(tr.get("eval_size", 0.2))
    min_pr_auc = float(tr.get("min_pr_auc", 0.8))

    train_idx, val_idx = train_test_split(
        df.index.values,
        test_size=test_size,
        random_state=int(tr.get("seed", 42)),
        stratify=y if len(set(y.tolist())) > 1 else None,
    )
    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)

    X_train = ext.fit_transform(df_train, prompt_col, resp_col)
    y_train = df_train[label_col].astype(int).values
    X_val = ext.transform(df_val, prompt_col, resp_col)
    y_val = df_val[label_col].astype(int).values

    lgb_params = tab.get("lgbm", {})
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=int(lgb_params.get("n_estimators", 400)),
        learning_rate=float(lgb_params.get("learning_rate", 0.05)),
        num_leaves=int(lgb_params.get("num_leaves", 64)),
        max_depth=int(lgb_params.get("max_depth", -1)),
        subsample=float(lgb_params.get("subsample", 0.85)),
        colsample_bytree=float(lgb_params.get("colsample_bytree", 0.85)),
        reg_alpha=float(lgb_params.get("reg_alpha", 0.0)),
        reg_lambda=float(lgb_params.get("reg_lambda", 1.0)),
        random_state=int(tr.get("seed", 42)),
        n_jobs=-1,
        verbose=-1,
        class_weight=lgb_params.get("class_weight", "balanced"),
    )
    clf.fit(X_train, y_train)
    val_prob = clf.predict_proba(X_val)[:, 1]
    pr_auc = float(average_precision_score(y_val, val_prob))
    status = "PASS" if pr_auc >= min_pr_auc else "FAIL"
    print(f"Validation PR-AUC={pr_auc:.4f} (target >= {min_pr_auc:.4f}) => {status}")

    out_dir = Path(paths["artifact_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "extractor": ext,
        "classifier": clf,
        "feature_names": ext.feature_names,
    }
    joblib.dump(bundle, out_dir / "tabular_bundle.joblib", compress=3)
    train_sha = hashlib.sha256(train_csv.resolve().read_bytes()).hexdigest()
    meta = {
        "backend": "tabular",
        "n_features": int(X_train.shape[1]),
        "n_train": len(df_train),
        "n_val": len(df_val),
        "val_pr_auc": pr_auc,
        "target_min_pr_auc": min_pr_auc,
        "meets_target": bool(pr_auc >= min_pr_auc),
        "train_csv": str(train_csv.resolve()),
        "train_csv_sha256": train_sha,
        "artifact": str((out_dir / "tabular_bundle.joblib").resolve()),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if pr_auc < min_pr_auc:
        raise ValueError(
            f"PR-AUC {pr_auc:.4f} ниже целевого порога {min_pr_auc:.4f}. "
            "Улучшите датасет/фичи/гиперпараметры."
        )
    print(f"Saved tabular model to {out_dir / 'tabular_bundle.joblib'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train LightGBM on tabular features")
    p.add_argument("--config", type=Path, default=repo_root() / "configs" / "default.yaml")
    args = p.parse_args()
    run_train_tabular(args.config)


if __name__ == "__main__":
    main()
