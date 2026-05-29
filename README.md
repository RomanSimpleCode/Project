# Restoration of Circuits

Проект для восстановления изображений технических схем и планов помещений из LQ в HQ. В пайплайне используются PyTorch, ClearML и Streamlit.

Проект покрывает полный цикл: подготовка пар HQ/LQ, создание ClearML Dataset, проверка загрузки данных, обучение CleanResidualUNet, логирование метрик/checkpoint-ов в ClearML, диагностические overfit-эксперименты и веб-интерфейс для инференса.

## Текущее состояние

В рабочем дереве нет файла `two_stage_one_file.py`, но `streamlit_app.py` импортирует из него `CONFIG` и `CleanResidualUNet`. Поэтому Streamlit-приложение в текущем виде не запустится, пока файл не будет восстановлен или импорт не будет переведён на `model.py`.

`model.py` сейчас является основным training-скриптом. Он не содержит resume-логики из внешнего ClearML/local checkpoint, а запускает обучение CleanResidualUNet с нуля и сохраняет новые checkpoint-ы.

## Структура проекта

| Файл/папка | Назначение |
|---|---|
| `create_clearml_dataset.py` | Создание пар `HQ`/`LQ` из архива, локальной папки или ClearML Dataset и загрузка результата в ClearML Dataset |
| `pyTorch_Dataset.py` | Загрузка ClearML Dataset, resize/padding/нормализация, создание train/test `DataLoader`, логирование примеров и артефактов |
| `model.py` | Основной скрипт обучения CleanResidualUNet v7 для восстановления LQ -> HQ |
| `streamlit_app.py` | Streamlit-интерфейс для восстановления изображения и сравнения с классическими методами; сейчас зависит от отсутствующего `two_stage_one_file.py` |
| `experiments/` | Диагностические overfit-эксперименты `diagnostic_overfit_unet_v1.py` ... `v7.py` |
| `floorplan_dataset/` | Локальный датасет: `HQ`, `LQ`, `clearml_download` |
| `clearml_pipeline_unified.html` | HTML-экспорт/описание ClearML pipeline |
| `train-00.tar.xz` | Локальный архив исходных данных |
| `pyproject.toml`, `uv.lock` | Описание Python-проекта и lock-файл зависимостей |

## Установка

Требуется Python `>=3.12`.

```bash
uv sync
```

Основные зависимости: `torch`, `torchvision`, `clearml`, `clearml-agent`, `streamlit`, `opencv-python-headless`, `pillow`, `numpy`, `matplotlib`, `pytorch-msssim`, `scikit-learn`, `plotly`, `pandas`, `joblib`, `tqdm`.

Для работы с ClearML должны быть настроены credentials. `model.py` может запросить `ClearML API access key` и `ClearML API secret key` через консоль, если они не найдены в окружении.

## 1. Создание датасета

`create_clearml_dataset.py` создаёт пары изображений:

- `HQ`: исходное изображение, сохранённое как PNG.
- `LQ`: версия HQ после случайного downscale/upscale и JPEG-сжатия.

Шум отдельно не добавляется.

Поддерживаемые источники:

- архив `.tar.xz`; из него потоково извлекаются PNG из папки `coco_vis`;
- локальная папка с изображениями `png`, `jpg`, `jpeg`, `bmp`;
- существующий ClearML Dataset по ID.

Пример:

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
|---|---|---|
| `--archive_path` | Путь к архиву `.tar.xz` | один из источников обязателен |
| `--local_folder` | Путь к локальной папке с изображениями | один из источников обязателен |
| `--clearml_dataset_id` | ID ClearML Dataset с исходными изображениями | один из источников обязателен |
| `--output_dir` | Папка для `HQ`, `LQ` и временных данных | `floorplan_dataset` |
| `--dataset_name` | Имя создаваемого ClearML Dataset | `FloorPlanCAD_HQ_LQ` |
| `--jpeg_quality_min` | Минимальное JPEG-качество для LQ | `10` |
| `--jpeg_quality_max` | Максимальное JPEG-качество для LQ | `50` |
| `--resolution_ratio_min` | Минимальный коэффициент downscale | `2` |
| `--resolution_ratio_max` | Максимальный коэффициент downscale | `4` |

Результат: локальные папки `HQ`/`LQ` и ID созданного ClearML Dataset.

## 2. Подготовка DataLoader

`pyTorch_Dataset.py` загружает ClearML Dataset, ищет папки `LQ` и `HQ`, приводит изображения к квадрату через resize с сохранением пропорций и padding, нормализует их и создаёт train/test `DataLoader`.

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
|---|---|---|
| `--dataset_id` | ID ClearML Dataset | обязательно |
| `--image_size` | Размер стороны квадратного изображения | `512` |
| `--norm_range` | Нормализация: `0_1` или `-1_1` | `0_1` |
| `--batch_size` | Размер батча | `16` |
| `--train_split` | Доля train-выборки | `0.8` |
| `--num_workers` | Количество worker-процессов | `0` |
| `--save_state` | Сохранить состояние DataLoader | `False` |
| `--state_dir` | Директория для состояния DataLoader | `.` |
| `--project_name` | Проект ClearML | `Image_Restoration` |
| `--task_name` | Имя задачи ClearML | генерируется автоматически |
| `--execute_remotely` | Запустить задачу через ClearML Agent | `False` |
| `--queue` | Очередь ClearML Agent | `default` |

Скрипт логирует статистику, несколько LQ/HQ-примеров и артефакт `dataloader_states` в ClearML.

## 3. Обучение модели

Основной training-скрипт: `model.py`.

Архитектура:

- `CleanResidualUNet`;
- residual learning: модель предсказывает correction, результат строится как `LQ + correction`;
- U-Net skip connections;
- блоки `ResConvBlock`, `ConvStage`, `UpBlock`;
- `GroupNorm`;
- обучение на патчах;
- выбор патчей с контролем доли содержимого, чтобы не обучаться только на пустом белом фоне.

Основная конфигурация в `model.py`:

| Параметр | Значение |
|---|---|
| ClearML project | `Vosstanovlenie_tehnicheskih_sistem` |
| ClearML task | `full_train_clean_resunet_v1.1` |
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
- `gain_line` и `gain_weighted` как улучшение относительно LQ.

Запуск:

```powershell
python model.py
```

Скрипт сохраняет:

- `best_model.pth`;
- `last_model.pth`;
- периодические `epoch_XXXX.pth`;
- визуализации validation samples;
- `summary.json`;
- ClearML-артефакты: `best_model`, `last_model`, epoch checkpoints, summary и outputs folder.

## 4. Streamlit-приложение

Запуск:

```powershell
uv run streamlit run streamlit_app.py
```

Адрес по умолчанию: `http://localhost:8501`.

Важно: текущий `streamlit_app.py` импортирует:

```python
from two_stage_one_file import CONFIG, CleanResidualUNet
```

Так как `two_stage_one_file.py` отсутствует в проекте, приложение нужно либо запускать после восстановления этого файла, либо изменить импорт на совместимый модуль с архитектурой и `CONFIG`, например `model.py`.

Что умеет приложение после исправления зависимости:

- загрузка LQ-изображения: `png`, `jpeg`, `jpg`, `bmp`, `tif`, `tiff`, `webp`;
- загрузка опционального HQ-эталона для расчёта метрик;
- загрузка модели из ClearML task `bc17547fbb28414c80122a671060c2f4`, artifact `best_model`;
- tiled inference с `tile_size=512` и `overlap=64`;
- восстановление пропорций результата после inference;
- сравнение с классическими методами;
- скачивание результата как `restored_image.png`.

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
python model.py

# 4. Запустить приложение после восстановления зависимости two_stage_one_file.py
# или после перевода streamlit_app.py на актуальный модуль модели
uv run streamlit run streamlit_app.py
```

Все ключевые задачи, метрики, checkpoint-ы и артефакты логируются в ClearML UI.
