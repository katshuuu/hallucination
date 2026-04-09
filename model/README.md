# Артефакты модели

После `./scripts/train.sh` веса CrossEncoder сохраняются в каталог **`cross_encoder/`** (см. `artifact_dir` в `configs/default.yaml`).

Для проверки без обучения можно скопировать готовые веса в `model/cross_encoder/` или выполнить полный цикл: `install` → `build_train_data` → `train`.
