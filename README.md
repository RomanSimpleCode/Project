# Restoration of Circuits

Пайплайн для восстановления изображений технических схем и планов помещений из LQ в HQ с использованием PyTorch, ClearML и Streamlit.

Проект покрывает полный цикл: подготовка пар HQ/LQ, создание ClearML Dataset, предобработка, обучение residual U-Net, продолжение обучения из checkpoint, сравнение с классическими методами и интерактивное восстановление изображения через веб-интерфейс.

## Структура проекта

| Файл/папка | Назначение |
|------------|------------|
| `create_clearml_dataset.py` | Создание пар `HQ`/`LQ` из архива, локальной папки или ClearML Dataset и загрузка результата в ClearML Dataset |
| `pyTorch_Dataset.py` | Загрузка ClearML Dataset, resize/padding/нормализация и создание train/test `DataLoader` |
| `model.py` | Скрипт полного обучения CleanResidualUNet без логики resume из внешнего ClearML checkpoint |
| `two_stage_one_file.py` | Основной training-скрипт CleanResidualUNet с возможностью продолжить обучение из локального checkpoint или артефакта ClearML |
| `streamlit_app.py` | Streamlit-интерфейс для восстановления изображений и сравнения с классическими методами |
| `experiments/` | Диагностические overfit-эксперименты `diagnostic_overfit_unet_v1.py` ... `v7.py` |
| `floorplan_dataset/` | Локальная структура датасета: `HQ`, `LQ`, `clearml_download` |
| `clearml_pipeline_unified.html` | HTML-экспорт/описание ClearML pipeline |
| `train-00.tar.xz` | Локальный архив исходных данных |

## Установка

Требуется Python `>=3.12`.

```bash
uv sync
```

Основные зависимости указаны в `pyproject.toml`: `torch`, `torchvision`, `clearml`, `clearml-agent`, `streamlit`, `opencv-python-headless`, `pillow`, `numpy`, `matplotlib`, `pytorch-msssim`, `scikit-learn`, `plotly`, `pandas`, `joblib`, `tqdm`.

Для работы с ClearML должны быть настроены credentials. Training-скрипты также могут запросить `ClearML API access key` и `ClearML API secret key` через консоль.

## 1. Создание датасета — `create_clearml_dataset.py`

Создаёт пары изображений:

- `HQ`: исходное изображение, сохранённое как PNG.
- `LQ`: версия HQ после downscale/upscale и JPEG-сжатия.

Шум специально не добавляется.

Поддерживаемые источники данных:

- архив `.tar.xz`; из него потоково извлекаются PNG из папки `coco_vis`;
- локальная папка с изображениями `png`, `jpg`, `jpeg`, `bmp`;
- существующий ClearML Dataset по ID.

```powershell
python create_clearml_dataset.py `
    --archive_path train-00.tar.xz `
    --output_dir floorplan_dataset `
    --dataset_name FloorPlanCAD_HQ_LQ `
    --jpeg_quality_min 10 `
    --jpeg_quality_max 50 `
    --resolution_ratio_min 2 `
    --resolution_ratio_max 4
```

Параметры:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--archive_path` | Путь к архиву `.tar.xz` | один из источников обязателен |
| `--local_folder` | Путь к локальной папке с изображениями | один из источников обязателен |
| `--clearml_dataset_id` | ID ClearML Dataset с исходными изображениями | один из источников обязателен |
| `--output_dir` | Папка для `HQ`, `LQ` и временных данных | `floorplan_dataset` |
| `--dataset_name` | Имя создаваемого ClearML Dataset | `FloorPlanCAD_HQ_LQ` |
| `--jpeg_quality_min` | Минимальное JPEG-качество для LQ | `10` |
| `--jpeg_quality_max` | Максимальное JPEG-качество для LQ | `50` |
| `--resolution_ratio_min` | Минимальный коэффициент downscale | `2` |
| `--resolution_ratio_max` | Максимальный коэффициент downscale | `4` |

Выход: локальные папки `HQ`/`LQ` и ID созданного ClearML Dataset.

## 2. Предобработка — `pyTorch_Dataset.py`

Скрипт загружает ClearML Dataset, ищет папки `LQ` и `HQ`, приводит изображения к квадрату через resize с сохранением пропорций и padding, нормализует их и создаёт train/test `DataLoader`.

```powershell
python pyTorch_Dataset.py `
    --dataset_id <DATASET_ID> `
    --image_size 512 `
    --norm_range 0_1 `
    --batch_size 16 `
    --train_split 0.8 `
    --num_workers 0 `
    --save_state
```

Параметры:

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--dataset_id` | ID датасета ClearML | обязательно |
| `--image_size` | Размер стороны квадрата | `512` |
| `--norm_range` | Нормализация: `0_1` или `-1_1` | `0_1` |
| `--batch_size` | Размер батча | `16` |
| `--train_split` | Доля train-выборки | `0.8` |
| `--num_workers` | Количество worker-процессов | `0` |
| `--save_state` | Сохранить состояние DataLoader | `False` |
| `--state_dir` | Директория для состояния DataLoader | `.` |
| `--project_name` | Проект ClearML | `Image_Restoration` |
| `--task_name` | Имя задачи ClearML | автогенерация |
| `--execute_remotely` | Запустить задачу через ClearML Agent | `False` |
| `--queue` | Очередь ClearML Agent | `default` |

## 3. Обучение модели

В проекте есть две близкие версии training-скрипта:

- `model.py` — полный train CleanResidualUNet.
- `two_stage_one_file.py` — актуальная версия с resume из ClearML/local checkpoint; именно её использует `streamlit_app.py` для архитектуры модели.

Архитектура:

- `CleanResidualUNet`;
- residual learning: модель предсказывает correction, итоговый результат строится как восстановленное изображение;
- U-Net skip connections;
- `ResConvBlock`, `ConvStage`, `UpBlock`;
- GroupNorm;
- tiled/patch-подход при обучении на патчах.

Основная конфигурация в `two_stage_one_file.py`:

| Параметр | Значение |
|----------|----------|
| ClearML project | `Vosstanovlenie_tehnicheskih_sistem` |
| ClearML task | `full_train_clean_resunet_v1.1_continue` |
| Dataset ID | `aa90ed3c5ca14bec9f9828a8891a5a59` |
| `image_size` | `1024` |
| `patch_size` | `512` |
| `test_split` | `0.20` |
| `batch_size` | `2` |
| `base_channels` | `64` |
| `epochs` | `150` |
| `lr` | `1e-4` |
| `early_stopping_patience` | `35` |
| `save_dir` | `full_train_clean_resunet_v7_outputs` |

Loss и метрики:

- `charbonnier`;
- line mask по тёмным линиям схемы;
- `line_l1`;
- `weighted_l1`;
- residual supervision;
- Sobel edge metric;
- `gain_line` как улучшение относительно LQ.

Запуск:

```powershell
python two_stage_one_file.py
```

Скрипт сохраняет:

- `best_model.pth`;
- `last_model.pth`;
- периодические `epoch_XXXX.pth`;
- визуализации validation samples;
- `summary.json`;
- артефакты в ClearML: `best_model`, `last_model`, epoch checkpoints, summary и outputs folder.

### Resume обучения

`two_stage_one_file.py` умеет продолжать обучение из:

- ClearML task artifact;
- локального `.pth` checkpoint.

Поля в `CONFIG`:

| Поле | Назначение |
|------|------------|
| `resume_from_task_id` | ID ClearML task, из которой скачать checkpoint |
| `resume_artifact_name` | Имя артефакта: обычно `best_model` или `last_model` |
| `resume_local_checkpoint` | Локальный путь к `.pth`; если указан, используется он |
| `resume_strict` | Строгая загрузка весов |
| `reset_optimizer_on_resume` | Сбросить optimizer/scheduler/scaler при resume |

## 4. Streamlit-приложение — `streamlit_app.py`

Интерфейс восстанавливает загруженное изображение через модель из ClearML и позволяет сравнить результат с классическими методами.

```powershell
uv run streamlit run streamlit_app.py
```

Адрес по умолчанию: `http://localhost:8501`.

Что умеет приложение:

- загрузка LQ-изображения: `png`, `jpeg`, `jpg`, `bmp`, `tif`, `tiff`, `webp`;
- загрузка опционального эталонного HQ для расчёта метрик;
- восстановление через `CleanResidualUNet`;
- tiled inference с `tile_size=512` и `overlap=64`;
- восстановление пропорций результата после inference;
- сравнение с классическими методами;
- скачивание результата как `restored_image.png`.

Параметры модели в текущем коде:

| Параметр | Значение |
|----------|----------|
| ClearML Task ID | `bc17547fbb28414c80122a671060c2f4` |
| Артефакт | `best_model` |
| Архитектура | `CleanResidualUNet` из `two_stage_one_file.py` |
| `TARGET_SIZE` | из `CONFIG["image_size"]`, сейчас `1024` |
| Tile size | `512` |
| Tile overlap | `64` |
| Base channels | из `CONFIG["base_channels"]`, сейчас `64` |

Классические методы сравнения:

- Lanczos resampling;
- Bicubic interpolation;
- Total Variation regularization;
- NEDI approximation через edge-preserving filter.

Метрики при наличии HQ-эталона:

- PSNR;
- SSIM;
- MSE;
- MAE;
- Line L1;
- Weighted L1;
- Gain Line;
- Edge ref.

## 5. Диагностические эксперименты

Папка `experiments/` содержит последовательность overfit-экспериментов:

```text
diagnostic_overfit_unet_v1.py
diagnostic_overfit_unet_v2.py
diagnostic_overfit_unet_v3.py
diagnostic_overfit_unet_v4.py
diagnostic_overfit_unet_v5.py
diagnostic_overfit_unet_v6.py
diagnostic_overfit_unet_v7.py
```

Они используются для проверки архитектурных идей на малом наборе изображений перед полным обучением.

## Пример полного цикла

```powershell
# 1. Создать HQ/LQ датасет и загрузить его в ClearML
python create_clearml_dataset.py `
    --archive_path train-00.tar.xz `
    --output_dir floorplan_dataset `
    --dataset_name FloorPlanCAD_HQ_LQ

# 2. Проверить загрузку и предобработку датасета
python pyTorch_Dataset.py `
    --dataset_id <DATASET_ID> `
    --image_size 512 `
    --batch_size 16

# 3. Запустить обучение
python two_stage_one_file.py

# 4. Запустить приложение
uv run streamlit run streamlit_app.py
```

Все ключевые задачи, метрики, checkpoints и артефакты логируются в ClearML UI.
