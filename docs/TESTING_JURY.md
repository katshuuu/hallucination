# Тестирование перед сдачей: скорость, память, ограничения

Цель — убедиться, что жюри сможет **воспроизвести** решение и что метрики **скорости** и **ресурсов** укладываются в ожидания. Актуальные лимиты (в т.ч. **500 ms** учитываемого времени, **A100 80GB**, правила **GigaChat forward** vs **градиенты**) — в **`docs/COMPETITION_RULES.md`**.

---

## 1. Быстрый смоук-тест (одна команда)

Из корня репозитория после `./scripts/install.sh` и (желательно) `./scripts/train.sh`:

```bash
chmod +x scripts/verify_solution.sh
./scripts/verify_solution.sh
```

Скрипт последовательно:

1. Проверяет контракт **`data/processed/train_merged.csv`** (`audit_train_csv`).
2. Запускает **`check_constraints.py`** (конфиг, отсутствие обязательных облачных ключей для инференса).
3. Если есть **`model/tabular/tabular_bundle.joblib`** — скорит пример бенча, выводит **`cli benchmark`**, затем **`measure_performance.py`** с батчем по `knowledge_bench_private.csv`.

Если модель ещё не обучена, шаги 3–5 пропускаются — сначала выполните `./scripts/train.sh`.

---

## 2. Ручная проверка по шагам

| Шаг | Команда | Ожидание |
|-----|---------|----------|
| Установка | `./scripts/install.sh` | venv, пакет `pip install -e .`, при необходимости кэш HF |
| Сборка train | `./scripts/build_train_data.sh` | `data/processed/train_merged.csv` |
| Аудит train | `python scripts/audit_train_csv.py` | `OK: ...` |
| Обучение | `./scripts/train.sh` | `model/tabular/tabular_bundle.joblib`, `train_meta.json` |
| Скоринг | `./scripts/score_private.sh` или `score_csv` | CSV с `hallucination_prob` |
| Ограничения | `python scripts/check_constraints.py` | нет обязательных API-ключей для скоринга |

---

## 3. Скорость (латентность)

### 3.1. Одна пара

```bash
python -m hallucination_detector.cli benchmark --repeats 50 --warmup 10
```

Смотрите **mean_ms** и **p95_ms**. Для сравнения с регламентом ориентируйтесь на **p95** или **mean** на целевом железе (у жюри может быть A100 или CPU — уточняйте).

### 3.2. Расширенный замер + батч

```bash
python scripts/measure_performance.py --repeats 50 --csv data/bench/knowledge_bench_private.csv
```

Выводит:

- задержку **на одну пару** (mean/p95 внутри `cli` для одиночного скорера);
- **ms_per_row** при последовательном проходе по CSV (реальный путь `score_csv`);
- грубую оценку **RSS** процесса до/после загрузки модели.

**Замечание:** первый запрос после старта процесса обычно медленнее (загрузка библиотек, прогрев). Для API используйте прогрев или смотрите метрики после `warmup`.

### 3.3. Снижение латентности

- В `configs/default.yaml`: `tabular.tfidf_max_features` (меньше — быстрее, хуже качество).
- Отключите **`use_lm_features`** если включали (distilgpt2 добавляет время).
- Убедитесь, что инференс идёт на **GPU**, если это предусмотрено (`torch.cuda.is_available()` в `check_constraints`).

---

## 4. Память

1. **`measure_performance.py`** — RSS процесса в МиБ (через `resource` или `psutil`, если установлен: `pip install psutil`).
2. При использовании GPU смотрите **`nvidia-smi`** во время `./scripts/train.sh` и во время скоринга.
3. Не храните в train CSV лишние тяжёлые колонки; батч на диске для скоринга читается **построчно через pandas** — при очень больших файлах можно позже перейти на chunked read (не обязательно для типичного бенча).

---

## 5. Ограничения соревнования (чек-лист для жюри)

| Критерий | Как проверить у себя |
|----------|----------------------|
| Нет внешних API на инференсе | `python scripts/check_constraints.py`; скоринг без интернета после кэша HF |
| Воспроизводимость | `./scripts/build_train_data.sh` → `./scripts/train.sh` → совпадение `train_meta.json` / SHA train |
| Текстовый train, не «анонимные фичи» | `python scripts/audit_train_csv.py`, см. `docs/END_TO_END_DATA.md` |
| Скоринг приватного формата | `score_csv` с колонками `prompt` + `response` |
| Зависимости зафиксированы | `requirements.txt`, `pyproject.toml` |

---

## 6. HTTP API (если сдаёте сервис)

```bash
./scripts/run_api.sh
# Другой терминал:
python scripts/test_api_client.py
```

Проверьте **GET /health/live**, **GET /health/ready** и **POST /predict** в Swagger (`/docs`). Латентность измеряйте клиентом с учётом сети localhost (в ответе `/predict` есть `inference_time_ms`).

---

## 7. Что приложить к отчёту (по желанию)

- Вывод `./scripts/verify_solution.sh` или лог `measure_performance.py`.
- Версия Python (`python -V`), версия CUDA / драйвера (если GPU).
- Краткая таблица: mean_ms / p95_ms на вашей машине и размер `tabular_bundle.joblib`.

Это снижает количество вопросов жюри по скорости и ресурсам.
