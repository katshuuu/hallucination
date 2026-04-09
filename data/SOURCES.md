# Источники строк в `train_merged.csv`

Файл **`processed/train_merged.csv`** собирается скриптом **`scripts/build_train_data.sh`** (`hallucination_detector.data.build`).

| Компонент | Где в коде | Поле `source` |
|-----------|------------|---------------|
| TruthfulQA (generation) | `data/hf_sources.py` → `load_truthful_qa` | `open_dataset` |
| SNLI | `load_snli` | `open_dataset` |
| PAWS labeled_final | `load_paws` | `open_dataset` |
| Шаблоны с эвристикой | `data/synthetic.py` | `synthetic` |
| Ручная разметка | `validation/manual_labeled_sample.csv` | `manual` |
| Доп. размеченные Q&A (уникальные пары; ИТ/ИБ/финансы/инженерия) | `supplemental/labeled_qa_diverse_v1.csv` | `supplemental` |
| Расширение supplemental (идемпотентно) | `scripts/append_corp300_to_supplemental.py` | — |

Препроцессинг текста при сборке HF: **`data/normalize.py`** (`clean_text`).

Нормализация тега источника: **`data/source_tags.py`**.

После сборки train содержит **только** тексты и метки (и id/source), без предвычисленных столбцов признаков — их даёт **`tabular/extractor.py`** при `train.sh`.
