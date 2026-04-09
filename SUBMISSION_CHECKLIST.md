# Submission Checklist

Короткий чек-лист перед отправкой решения жюри.

## 1) Что именно обучает модель сейчас

Модель обучается на **текстовых данных** в файле:

- `data/processed/train_merged.csv`

Формат строк:

- `id, prompt, response, label, source`

Где:

- `label=1` — галлюцинация / фактическая ошибка
- `source` — происхождение строки: `open_dataset | synthetic | manual | judge_labeled`

Откуда берутся эти строки:

- Открытые датасеты (через HF): `src/hallucination_detector/data/hf_sources.py`
- Синтетика: `src/hallucination_detector/data/synthetic.py`
- Ручная разметка: `data/validation/manual_labeled_sample.csv`
- Сборка и унификация: `src/hallucination_detector/data/build.py`
- Манифест источников: `data/SOURCES.md`

Принципиально важно: модель **не** обучается на «анонимных числовых фичах». Признаки всегда восстанавливаются из `prompt/response` в `src/hallucination_detector/tabular/extractor.py`.

## 2) Что приложить в сдачу

- [ ] Текстовый train: `data/processed/train_merged.csv`
- [ ] Код сборки train: `scripts/build_train_data.sh` + `src/hallucination_detector/data/*`
- [ ] Код обучения: `scripts/train.sh` + `src/hallucination_detector/tabular/train.py`
- [ ] Код скоринга приватного теста: `scripts/score_private.sh` / `hallucination_detector.score_csv`
- [ ] Зависимости: `requirements.txt`, `pyproject.toml`
- [ ] README с инструкцией запуска: `README.md`

## 3) Команды перед отправкой

```bash
cd /Users/katshu/sber
source .venv/bin/activate

# (опционально) пересобрать train из источников
./scripts/build_train_data.sh

# проверить, что train корректный и без утечки public benchmark
python scripts/audit_train_csv.py
python scripts/check_no_public_benchmark_overlap.py

# обучить модель (есть порог PR-AUC >= 0.8 в конфиге)
./scripts/train.sh

# смоук-проверка решения
./scripts/verify_solution.sh

# скоринг приватного CSV
./scripts/score_private.sh
```

## 4) Контроль ограничений и качества

- [ ] В инференсе нет внешних API для факт-чекинга
- [ ] **Не** используете публичный preview-бенчмарк как **обучающий** набор (только для отладки); нет пересечений train с `knowledge_bench_public.csv` при `allow_public_overlap: false`
- [ ] `training.min_pr_auc` выполняется (см. `model/tabular/train_meta.json`); итоговая оценка жюри — **PR-AUC**, при равенстве — **скорость**
- [ ] **Время детекции** замерено в пайплайне (`POST /predict`: `wall_time_ms`, `billable_time_ms`; CLI `benchmark` / `measure_performance.py`). Лимит **500 ms** по учитываемому времени; эталон **1× A100 80GB** — см. **`docs/COMPETITION_RULES.md`**
- [ ] Скорость/память измерены (`docs/TESTING_JURY.md`, `scripts/measure_performance.py`)

## 5) Как обучить модель ещё лучше

Ниже практичный roadmap с максимальным эффектом на PR-AUC.

1. **Улучшить данные (самый сильный рычаг)**
   - Увеличить долю `open_dataset` и `manual` относительно простой синтетики.
   - Добавить сложные негативы: «правдоподобная, но неверная фактология», частичные ошибки, даты/числа/имена.
   - Балансировать `source` и классы (чтобы модель не учила «стиль источника» вместо качества ответа).

2. **Сильнее anti-leakage и дедуп**
   - Делать дедуп не только по точному тексту, но и по нормализованным/почти-дубликатам (fuzzy near-duplicates).
   - Контролировать пересечения между train/val по кластерам похожих пар.

3. **Качественнее валидация**
   - Перейти с single split на стратифицированный K-fold (например, 5-fold) и выбирать конфиг по среднему PR-AUC.
   - Вести отдельный «hard validation split» (самые сложные источники и примеры).

4. **Фичи**
   - Тонкая настройка `tfidf_max_features`, n-grams, min_df/max_df.
   - Добавить дополнительные lexical-fact признаки: несогласованность чисел, дат, единиц измерения.
   - Включить `use_lm_features: true` (NLL/PPL) и замерить компромисс PR-AUC vs latency.

5. **Модель**
   - Пробовать CatBoost/XGBoost рядом с LightGBM и/или ансамбль 2-3 моделей.
   - Калибровать вероятности (Platt / isotonic) на holdout для стабильного ранжирования.

6. **Judge-features (осторожно)**
   - Добавлять `judge_*` только как дополнительный слой, не как единственный источник истины.
   - Сохранять сырой judge-output и скрипт генерации для прозрачности.

## 6) Что показать жюри в отчёте

- Доказательство воспроизводимости: команды из раздела 3.
- `train_meta.json` (включая `train_csv_sha256` и `val_pr_auc`).
- Краткая таблица метрик скорости/памяти из `scripts/measure_performance.py`.
- Краткое обоснование, почему выбранный набор источников и фич даёт новизну/качество.

