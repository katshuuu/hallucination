from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sentence_transformers import InputExample
from sentence_transformers.cross_encoder import CrossEncoder
from torch.utils.data import DataLoader

from hallucination_detector.columns_util import response_column
from hallucination_detector.settings import load_config, repo_root


def build_examples(
    df: pd.DataFrame,
    prompt_col: str,
    response_col: str,
    label_col: str,
    num_labels: int,
) -> list[InputExample]:
    examples: list[InputExample] = []
    for _, row in df.iterrows():
        label = row[label_col]
        if num_labels == 1:
            y = float(label)
        else:
            y = int(label)
        examples.append(
            InputExample(
                texts=[str(row[prompt_col]), str(row[response_col])],
                label=y,
            )
        )
    return examples


def run_train(config_path: Path) -> None:
    cfg = load_config(config_path)
    m = cfg["model"]
    tr = cfg["training"]
    cols = cfg["columns"]
    paths = cfg["paths"]

    train_csv = Path(paths["train_csv"])
    if not train_csv.is_file():
        raise FileNotFoundError(
            f"Train CSV not found: {train_csv}. Сначала: ./scripts/build_train_data.sh"
        )

    df = pd.read_csv(train_csv)
    prompt_col = cols["prompt"]
    resp_col = response_column(df, cols)
    label_col = cols["label"]
    num_labels = int(m.get("num_labels", 1))

    base_model = tr.get("base_model") or m.get("name")
    if not base_model:
        raise ValueError("Задайте training.base_model или model.name в конфиге.")

    rng = tr.get("seed", 42)
    df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)

    train_examples = build_examples(df, prompt_col, resp_col, label_col, num_labels)
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=16)

    artifact = Path(paths.get("artifact_dir") or "model/cross_encoder")
    artifact.parent.mkdir(parents=True, exist_ok=True)

    model = CrossEncoder(
        base_model,
        max_length=int(m.get("max_length", 256)),
        num_labels=num_labels,
    )

    epochs = int(tr.get("epochs", 3))
    lr = float(tr.get("learning_rate", 2e-5))
    warmup_ratio = float(tr.get("warmup_ratio", 0.1))
    steps_per_epoch = max(1, len(train_loader))
    warmup_steps = int(epochs * steps_per_epoch * warmup_ratio)

    model.fit(
        train_dataloader=train_loader,
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": lr},
        output_path=str(artifact),
        show_progress_bar=True,
    )
    print(f"Saved CrossEncoder to {artifact}")


def main() -> None:
    p = argparse.ArgumentParser(description="Train CrossEncoder for hallucination scoring")
    p.add_argument(
        "--config",
        type=Path,
        default=repo_root() / "configs" / "default.yaml",
        help="Path to YAML config",
    )
    args = p.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
