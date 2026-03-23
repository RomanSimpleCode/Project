# Restoration of Circuits

Пайплайн для восстановления схем из сжатых изображений с использованием ClearML.

## Структура пайплайна

1. **data_preparation** (`create_clearml_dataset.py`) — создание пар LQ/HQ изображений и загрузка в ClearML Dataset
2. **preprocessing** (`pyTorch_Dataset.py`) — предобработка изображений (resize, нормализация) и создание PyTorch DataLoaders

## Установка

```bash
uv sync
```

## Запуск скриптов

### 1. Data Preparation — `create_clearml_dataset.py`

Создаёт пары LQ/HQ изображений из исходных данных и загружает в ClearML Dataset.

LQ создаётся из HQ путём:
- Снижения JPEG качества
- Снижения разрешения с последующим растягиванием

**Источники данных** (указать один):
- Архив `.tar.xz` с изображениями
- Локальная папка с изображениями
- ClearML Dataset по ID

```bash
python create_clearml_dataset.py \
    --archive_path <путь_к_архиву.tar.xz> \
    --output_dir floorplan_dataset \
    --dataset_name FloorPlanCAD_HQ_LQ \
    --jpeg_quality_min 10 \
    --jpeg_quality_max 50 \
    --resolution_ratio_min 2 \
    --resolution_ratio_max 4
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--archive_path` | Путь к архиву `.tar.xz` с исходными изображениями | — |
| `--local_folder` | Путь к локальной папки с изображениями (png, jpg, jpeg, bmp) | — |
| `--clearml_dataset_id` | ID ClearML Dataset для загрузки исходных изображений | — |
| `--output_dir` | Папка для выходных данных | `floorplan_dataset` |
| `--dataset_name` | Имя датасета ClearML | `FloorPlanCAD_HQ_LQ` |
| `--jpeg_quality_min` | Минимальное качество JPEG для LQ | `10` |
| `--jpeg_quality_max` | Максимальное качество JPEG для LQ | `50` |
| `--resolution_ratio_min` | Мин. коэффициент снижения разрешения (1/X) | `2` |
| `--resolution_ratio_max` | Макс. коэффициент снижения разрешения (1/X) | `4` |

**Выход:** Dataset ID ClearML (используется в следующем этапе)

---

### 2. Preprocessing — `pyTorch_Dataset.py`

Загружает датасет из ClearML, применяет предобработку и создаёт DataLoaders для train/test.

```bash
python pyTorch_Dataset.py \
    --dataset_id <ID_из_предыдущего_этапа> \
    --image_size 512 \
    --norm_range 0_1 \
    --batch_size 16 \
    --train_split 0.8 \
    --num_workers 0 \
    --save_state
```

**Параметры:**

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--dataset_id` | ID датасета ClearML из этапа 1 | **обязательно** |
| `--image_size` | Размер стороны квадрата (resize + padding) | `512` |
| `--norm_range` | Диапазон нормализации: `0_1` или `-1_1` | `0_1` |
| `--batch_size` | Размер батча | `16` |
| `--train_split` | Доля train выборки | `0.8` |
| `--num_workers` | Количество процессов для загрузки данных | `0` |
| `--save_state` | Сохранить состояние DataLoader для воспроизводимости | `False` |
| `--state_dir` | Директория для сохранения состояния | `.` |
| `--project_name` | Имя проекта ClearML | `Image_Restoration` |
| `--task_name` | Имя задачи ClearML | автогенерация |
| `--execute_remotely` | Выполнить удалённо через ClearML Agent | `False` |
| `--queue` | Очередь для удалённого выполнения | `default` |

**Выход:**
- Train/Test DataLoaders
- Артефакты в ClearML (состояния DataLoaders, метаданные)
- Примеры изображений в ClearML UI

---

## Пример полного пайплайна

```bash
# Этап 1: Создание датасета
python create_clearml_dataset.py \
    --archive_path floorplan_dataset/train-00.tar.xz \
    --jpeg_quality_min 10 \
    --jpeg_quality_max 50 \
    --resolution_ratio_min 2 \
    --resolution_ratio_max 4

# Этап 2: Предобработка (использовать Dataset ID из вывода этапа 1)
python pyTorch_Dataset.py \
    --dataset_id <DATASET_ID> \
    --image_size 512 \
    --batch_size 16
```

Все задачи и артефакты отслеживаются в ClearML UI.
