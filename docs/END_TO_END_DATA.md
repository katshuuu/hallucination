# End-to-end: тексты → препроцессинг → признаки → модель

Этот документ для проверяющего: **обучение не строится на анонимных числовых векторах**. Единственный обязательный вход в обучение — **таблица с текстами** `prompt` и `response` плюс метка и происхождение.

## Что считается «каноническим» train

Файл **`data/processed/train_merged.csv`** (или тот же формат после `./scripts/build_train_data.sh`) должен содержать как минимум:

| Колонка | Назначение |
|---------|------------|
| `id` | Идентификатор строки |
| `prompt` | Исходный текст вопроса/контекста |
| `response` | Исходный текст ответа |
| `label` | 0/1 |
| `source` | `open_dataset` \| `synthetic` \| `manual` \| `judge_labeled` |

**Запрещено** подавать в `train.sh` CSV, в котором **нет** колонок `prompt` и `response` с текстом — такой файл не пройдёт проверку `hallucination_detector.data.dataset_contract`.

Признаки для LightGBM **всегда** вычисляются в коде из `(prompt, response)`:

1. `src/hallucination_detector/data/normalize.py` — нормализация пробелов/длины (используется при сборке HF-данных).
2. `src/hallucination_detector/tabular/extractor.py` — TF-IDF, статистики текста, эмбеддинги, опционально LM и judge-колонки.

То есть восстановить признаки можно, имея **только тексты** и тот же код/конфиг.

## Как воспроизвести train с нуля

1. Исходники и скрипты: `configs/data_pipeline.yaml`, `src/hallucination_detector/data/build.py`, `hf_sources.py`, `synthetic.py`, `data/validation/manual_labeled_sample.csv`.
2. Сборка: `./scripts/build_train_data.sh` → `data/processed/train_merged.csv`.
3. Обучение: `./scripts/train.sh` → `model/tabular/tabular_bundle.joblib`.

## Проверка отсутствия «подмешивания» приватного теста

- Приватный тест организаторы подают **без меток**; в train **не должны** попадать те же пары `(prompt, response)`, что в приватном сете, если вы их туда не копировали.
- Ревьюер может сравнить **хеши текстов** train с известным публичным бенчем и убедиться, что приватный CSV не включён в `train_merged.csv` как текстовые поля.
- Скрипт **`scripts/reproduce_features_from_csv.py`** показывает, что вектор признаков однозначно получается из переданных строк (см. `--print-hash`).

## Публичный бенч

`knowledge_bench_public.csv` в корне — только для отладки; **не** используйте его как единственный train (правила соревнования).
