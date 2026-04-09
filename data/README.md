# Данные и происхождение разметки

**Важно:** обучение идёт **только** из CSV с исходными текстами `prompt` и `response`. Отдельного «анонимного» train из одних чисел в репозитории нет; признаки всегда восстанавливаются кодом из текстов (см. **`docs/END_TO_END_DATA.md`**, **`data/SOURCES.md`**).

Проверка контракта train-файла: `python scripts/audit_train_csv.py`.

Воспроизведение вектора признаков из строки текста (нужен уже обученный `model/tabular/`): `python scripts/reproduce_features_from_csv.py --print-hash`.

## Целевая схема строки

| Поле | Описание |
|------|-----------|
| `id` | Уникальный идентификатор (UUID или ручной id из `validation/manual_labeled_sample.csv`) |
| `prompt` | Вопрос / контекст |
| `response` | Ответ модели |
| `label` | 0 — допустимо, 1 — галлюцинация / фактическая ошибка |
| `source` | `open_dataset` \| `synthetic` \| `manual` \| `judge_labeled` |

Нормализация сырых тегов — `src/hallucination_detector/data/source_tags.py`.

## Путь данных

1. **Источники** — `configs/data_pipeline.yaml`, код `src/hallucination_detector/data/`: TruthfulQA, SNLI, PAWS, синтетика, ручной CSV.
2. **Препроцессинг** — `normalize.py`.
3. **Признаки** — `src/hallucination_detector/tabular/extractor.py` (слои 1–4; judge только если колонки есть в таблице).
4. **Обучение** — LightGBM, `scripts/train.sh`.
5. **Скоринг** — `scripts/score_private.sh`.

## Слой judge (опционально)

Числовые колонки `judge_relevance`, `judge_factuality`, `judge_consistency`, `judge_completeness` можно получить отдельным скриптом (например `scripts/llm_judge_optional.py` с сохранением сырого JSONL). Их нужно **смержить** в train CSV до `train.sh`. Не используйте только judge как разметку.

## Публичный бенч

`knowledge_bench_public.csv` — только отладка, не основной train.
