# Детектор галлюцинаций в ответах (табличные признаки + LightGBM)

Репозиторий содержит **полный конвейер**: сбор текстового обучающего датасета, извлечение признаков из пар `(prompt, response)`, обучение классификатора, скоринг CSV без меток и опциональный HTTP API. На инференсе **не используются внешние API** (только локальные модели и кэш Hugging Face после установки).

---

## 1. Роль данных и правила соревнования

### 1.1. Публичный бенч и оценка своего решения

Сопровождательный датасет (в т.ч. публичный бенч вроде `knowledge_bench_public.csv`, если он есть в репозитории) организаторы подают **для оценки качества вашего подхода** и понимания, насколько вы приблизились к целевой метрике. Это **не** приглашение обучать финальную модель только на этом наборе.

В бейзлайне могли показать обучение модели на тестовой выборке **как иллюстрацию ошибки** — так делать **не нужно**: это и даёт завышенные оценки на публичной части, и плохо переносится на реальный тест.

**Рекомендуемое использование публичного бенча:** отладка пайплайна, подбор архитектуры, грубая оценка PR-AUC/латентности — **не** как единственный источник разметки для production-модели.

Дополнительно в проекте есть автоматическая защита:

- `scripts/check_no_public_benchmark_overlap.py` проверяет пересечения train с `knowledge_bench_public.csv`;
- `./scripts/train.sh` завершится ошибкой, если найдены совпадающие пары `(prompt, response)` и `training.allow_public_overlap: false`.

### 1.2. Сбор обучающего датасета — часть задачи

Вы **сами собираете** обучающий корпус. Допустимы:

- открытые датасеты (например, через Hugging Face в скрипте сборки);
- синтетическая генерация пар и эвристическая разметка;
- слабая разметка через **LLM-as-judge** (с сохранением текстов и скрипта, см. `scripts/llm_judge_optional.py`);
- ручная разметка фрагментов (`data/validation/manual_labeled_sample.csv`).

В репозитории задан **воспроизводимый пример** пайплайна: `configs/data_pipeline.yaml` + `src/hallucination_detector/data/` → `data/processed/train_merged.csv`. Источники строк описаны в **`data/SOURCES.md`**.

### 1.3. Приватный тест в день сдачи

В день сдачи вы получите **приватный CSV без меток** (колонки `prompt`, `response` — или совместимые имена, см. ниже). Нужно:

1. проскорить файл и получить колонку со скором (например, `hallucination_prob`);
2. приложить результат к решению.

Организаторы автоматически посчитают метрику по всем участникам, отберут топ и **уже отобранные решения** проверят вручную на **воспроизводимость** и **отсутствие обучения на приватных сэмплах**.

Поэтому в решении должны быть явно:

| Требование | В этом репозитории |
|------------|-------------------|
| Датасет, на котором строилась модель | `data/processed/train_merged.csv` (+ скрипт пересборки `scripts/build_train_data.sh`) |
| Код воспроизведения (обучение / приложение) | `scripts/train.sh`, `src/hallucination_detector/tabular/`, опционально API `main.py` |
| Код скоринга приватного теста | `scripts/score_private.sh`, модуль `hallucination_detector.score_csv` |
| Список зависимостей | `pyproject.toml`, `requirements.txt` |

---

## 2. Как устроено решение

### 2.1. Идея модели

Для каждой пары **вопрос + ответ** модель выдаёт **вероятность галлюцинации** (положительный класс для PR-AUC: фактическая ошибка / несоответствие / «плохой» ответ в смысле метки train).

Используется **табличный стек**:

1. **Слой 1:** дешёвые текстовые статистики + отдельные **TF-IDF** для `prompt`, `response` и их конкатенации.
2. **Слой 2:** эмбеддинги предложений (**`all-MiniLM-L6-v2`**), косинус, разность и поэлементное произведение векторов.
3. **Слой 3 (опционально):** локальный **distilgpt2** — NLL и PPL по тексту ответа (`tabular.use_lm_features` в `configs/default.yaml`).
4. **Слой 4 (опционально):** числовые колонки `judge_*` в CSV, если вы их добавили отдельным скриптом.

Классификатор: **LightGBM**. Признаки **никогда не хранятся в train CSV как единственный сигнал**: в CSV лежат **исходные тексты** и метка; вектор признаков считается в коде (`src/hallucination_detector/tabular/extractor.py`). Контракт проверяется при обучении (`hallucination_detector.data.dataset_contract`).

Подробнее: **`docs/END_TO_END_DATA.md`**, **`data/README.md`**.

### 2.2. Структура каталогов

```
configs/           # default.yaml — обучение и инференс; data_pipeline.yaml — сборка train
data/
  processed/       # train_merged.csv (тексты + label + source)
  validation/      # ручная разметка (пример)
model/             # артефакты после train (в .gitignore по умолчанию)
scripts/           # install, build_train_data, train, score_private, run_api, audit, …
src/hallucination_detector/
  data/            # сборка датасета, нормализация, контракт CSV
  tabular/         # признаки, обучение LGBM, скорер
  api/             # FastAPI (опционально)
main.py            # uvicorn main:app
```

---

## 3. Установка

```bash
chmod +x scripts/*.sh
./scripts/install.sh
```

Создаётся виртуальное окружение `.venv`, ставятся зависимости из `requirements.txt` и пакет в режиме editable (`pip install -e .`).

Переменная **`SKIP_PRELOAD=1`** при `install.sh` отключает предзагрузку весов в кэш Hugging Face.

---

## 4. Полный цикл: данные → обучение → скоринг

### 4.1. Сборка обучающего CSV

```bash
./scripts/build_train_data.sh
```

Нужен доступ к сети для загрузки датасетов с Hugging Face (TruthfulQA, SNLI, PAWS). Для **воспроизводимой сборки без сети** используйте флаг **`--offline`** (или в `configs/data_pipeline.yaml` выставьте `offline_only: true`) — в train попадут синтетика, manual и **`data/supplemental/labeled_qa_diverse_v1.csv`** (дополнительные размеченные пары; расширение: `scripts/append_corp300_to_supplemental.py`, идемпотентно):

```bash
./scripts/build_train_data.sh --offline
```

После сборки выполняется проверка контракта текстового датасета; результат — **`data/processed/train_merged.csv`**.

Дополнительно вручную:

```bash
python scripts/audit_train_csv.py
```

### 4.2. Обучение

```bash
./scripts/train.sh
```

Во время обучения считается holdout-метрика **PR-AUC** (по умолчанию `training.eval_size: 0.2`).
Целевой порог задаётся в `configs/default.yaml`:

```yaml
training:
  min_pr_auc: 0.8
```

Если `val_pr_auc < min_pr_auc`, обучение завершится ошибкой (жёсткая проверка качества).

Сохраняется:

- **`model/tabular/tabular_bundle.joblib`** — экстрактор + LightGBM;
- **`model/tabular/train_meta.json`** — число признаков, размеры train/val, `val_pr_auc`, целевой порог, путь к train CSV и **SHA-256** файла train (для сверки у проверяющего).

### 4.3. Скоринг приватного / бенч CSV

Файл с колонками **`prompt`** и **`response`** (или **`model_answer`** / **`answer`** — подхватятся автоматически):

```bash
./scripts/score_private.sh
```

По умолчанию читает `data/bench/knowledge_bench_private.csv`, пишет `data/bench/knowledge_bench_private_scores.csv` с колонкой **`hallucination_prob`**.

Произвольные пути:

```bash
source .venv/bin/activate
python -m hallucination_detector.score_csv \
  --config configs/default.yaml \
  --input путь/к/приватному.csv \
  --output путь/к/результату.csv
```

Переопределение каталога с артефактами:

```bash
python -m hallucination_detector.score_csv ... --model путь/к/каталогу/с/tabular_bundle.joblib
```

---

## 5. Воспроизводимость и проверка «честности» данных

- **`scripts/reproduce_features_from_csv.py`** — показать, что признаки получаются из текстов строки train (нужен уже обученный bundle).
- **`docs/END_TO_END_DATA.md`** — логика end-to-end без анонимных только-числовых train-файлов.
- Публичный бенч в корне репозитория используйте для **оценки своего решения**, а не как единственный обучающий сигнал.

Правила жюри по **времени детекции (500 ms)**, **GPU A100**, **PR-AUC / скорость при равенстве**, **GigaChat forward / градиенты** и запрету внешних API на инференсе — в **`docs/COMPETITION_RULES.md`**.

---

## 6. Зависимости

См. **`requirements.txt`** и **`pyproject.toml`**. Основное:

- `torch`, `sentence-transformers`, `transformers`, `lightgbm`, `scipy`, `scikit-learn`, `pandas`, `numpy`, `pyyaml`, `datasets`, `tqdm`;
- опционально API: `fastapi`, `uvicorn[standard]`;
- опционально LLM-judge: `pip install openai` (см. `[project.optional-dependencies]` в `pyproject.toml`).

---

## 7. HTTP API (опционально)

После обучения модели:

```bash
source .venv/bin/activate
pip install -e .
uvicorn main:app --host 127.0.0.1 --port 8000
```

Или `./scripts/run_api.sh --reload`. Swagger: **http://127.0.0.1:8000/docs** — ручка **`POST /predict`**.

Эндпоинты для эксплуатации: **`GET /health/live`** (liveness), **`GET /health/ready`** (readiness, 503 если модель не загружена), **`GET /version`**. Заголовок **`X-Request-ID`** пробрасывается и возвращается в ответе. Ответ **`POST /predict`** включает замер времени: **`wall_time_ms`**, **`billable_time_ms`** (учитываемое по правилам жюри), **`jury_max_billable_ms`**, а также **`inference_time_ms`** (= wall, для совместимости). Настройки — **`configs/default.yaml`** → **`timing`**. CORS: **`HALLUCINATION_CORS_ORIGINS`** (список через запятую).

Схемы запроса/ответа: `src/hallucination_detector/api/schemas.py`. Конфиг: переменная **`HALLUCINATION_CONFIG`** (по умолчанию `configs/default.yaml`). Поддерживается **`backend: tabular`**.

Проверка клиентом без браузера: `python scripts/test_api_client.py`.

### Веб-интерфейс (загрузка CSV)

После запуска сервера откройте в браузере **`http://127.0.0.1:8000/ui`**: можно загрузить CSV с колонками `prompt` и `response` (или `model_answer` / `answer`), получить превью скоров и метрики (время обработки, распределение вероятностей; при наличии колонки `label` — PR-AUC и accuracy @0.5).

### Docker

После **`./scripts/train.sh`** (артефакты в `model/tabular/`):

```bash
docker compose up --build
```

Сервис слушает порт **8000**, каталог с моделью монтируется read-only (см. `docker-compose.yml`).

---

## 8. Как протестировать решение (скорость, память, ограничения)

Полное руководство: **`docs/TESTING_JURY.md`**.

**Автоматический смоук-тест** (после `install` и желательно `train`):

```bash
chmod +x scripts/verify_solution.sh
./scripts/verify_solution.sh
```

Проверяется: контракт train-CSV, конфиг и отсутствие обязательных облачных ключей для инференса, скоринг примера, латентность (`cli benchmark`), детальный замер `measure_performance.py` (в т.ч. **ms на строку** для всего бенч-файла).

**Отдельные команды:**

| Задача | Команда |
|--------|---------|
| Ограничения (API, CUDA, наличие модели) | `python scripts/check_constraints.py` |
| Латентность одной пары | `python -m hallucination_detector.cli benchmark --repeats 50` |
| Латентность + память RSS + батч по CSV | `python scripts/measure_performance.py --csv data/bench/knowledge_bench_private.csv` |

Ориентир по времени — **сотни миллисекунд на пару** на целевом железе (уточняйте регламент); смотрите **mean** и **p95** в выводе. Уменьшить время можно за счёт `tabular.tfidf_max_features` и отключения `use_lm_features` в `configs/default.yaml`.

---

## 9. CLI и метрики

```bash
python -m hallucination_detector.cli benchmark   # латентность одной пары
```

Целевая метрика соревнования — **PR-AUC**; латентность на инференсе — ориентир порядка **сотен миллисекунд** на пару (зависит от GPU/CPU и конфига).

---

## 10. Альтернатива: CrossEncoder

В коде остаётся путь **`backend: cross_encoder`** и скрипт **`scripts/train_cross_encoder.sh`** (дообучение `CrossEncoder`). Для сдачи по умолчанию используется **табличный стек** (`configs/default.yaml`).

---

## 11. Чек-лист перед отправкой решения

- [ ] В репозитории есть **текстовый train** и описание/скрипт его получения (`data/processed/`, `build_train_data.sh`).
- [ ] Есть **обучение** (`train.sh`) и **артефакты** описаны (куда кладётся `tabular_bundle.joblib`).
- [ ] Есть **скоринг без меток** (`score_private.sh` или `score_csv`).
- [ ] Указаны **зависимости** (`requirements.txt` / `pyproject.toml`).
- [ ] Вы **не** используете приватный тест как train и **не** подменяете задачу обучением только на публичном тесте как на «истинном» train.

При вопросах проверяющих по воспроизводимости опирайтесь на **`train_meta.json`**, **`audit_train_csv.py`** и **`docs/END_TO_END_DATA.md`**.
# hallucination
