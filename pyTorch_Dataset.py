"""
Второй этап обработки датасета для восстановления схем.

Загружает датасет из ClearML (созданный на первом этапе create_clearml_dataset.py),
применяет предобработку (resize до квадрата, нормализация) и создаёт DataLoader
для train/test с возможностью сохранения/загрузки состояния.

Входные параметры:
    dataset_id (str): ID датасета из Task 1 — pipeline передаёт его автоматически
    image_size (int): Размер стороны (500, 1000 или 2000 и т.д.). Все изображения 
                      приводятся к квадрату этого размера. Если изображение не 
                      квадратное, добавляется padding (0) до квадрата.
    norm_range (str): "0_1" — нормализация в [0,1] или "-1_1" — в [-1,1]

Что выдаёт:
    train_dataloader, test_dataloader
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import argparse
from pathlib import Path
from datetime import datetime
from clearml import Task, Dataset as ClearMLDataset
import sys
if sys.platform == "win32":
    Task.ignore_requirements("pywin32")
from typing import Tuple, Optional, Dict, Any


# -----------------------------
# Функции предобработки изображений
# -----------------------------

def pad_to_square(image: torch.Tensor, target_size: int) -> torch.Tensor:
    """
    Добавляет padding к изображению до квадратного размера target_size x target_size.
    Padding заполняется нулями.
    
    Args:
        image: Тензор изображения формы (C, H, W)
        target_size: Целевой размер стороны
        
    Returns:
        Квадратный тензор формы (C, target_size, target_size)
    """
    c, h, w = image.shape
    
    if h == w == target_size:
        return image
    
    # Вычисляем padding
    pad_h = target_size - h
    pad_w = target_size - w
    
    # Padding: (left, right, top, bottom)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    
    return torch.nn.functional.pad(image, padding, mode='constant', value=0.0)


def normalize_image(image: torch.Tensor, norm_range: str = "0_1") -> torch.Tensor:
    """
    Нормализует изображение в указанный диапазон.
    
    Args:
        image: Тензор изображения формы (C, H, W) со значениями в [0, 1]
        norm_range: "0_1" для [0, 1] или "-1_1" для [-1, 1]
        
    Returns:
        Нормализованный тензор
    """
    if norm_range == "0_1":
        return image
    elif norm_range == "-1_1":
        return image * 2.0 - 1.0
    else:
        raise ValueError(f"Неподдерживаемый norm_range: {norm_range}. Используйте '0_1' или '-1_1'")


def preprocess_image(
    path: Path, 
    img_size: int, 
    norm_range: str = "0_1"
) -> torch.Tensor:
    """
    Предобработка изображения:
    1. Чтение изображения
    2. Конвертация BGR → RGB
    3. Resize с сохранением пропорций и padding до квадрата
    4. Нормализация в [0, 1] или [-1, 1]
    5. Конвертация в тензор (C, H, W)
    
    Args:
        path: Путь к изображению
        img_size: Размер стороны квадрата
        norm_range: Диапазон нормализации
        
    Returns:
        Тензор изображения формы (3, img_size, img_size)
    """
    # Чтение изображения
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    
    # Конвертация BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Получение исходных размеров
    h, w = img.shape[:2]
    
    # Вычисление масштаба для вписывания в target_size
    scale = min(img_size / w, img_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize с сохранением пропорций
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Конвертация в float и нормализация в [0, 1]
    img = img.astype('float32') / 255.0
    
    # Конвертация в тензор (C, H, W)
    img_tensor = torch.tensor(img).permute(2, 0, 1)
    
    # Padding до квадрата
    img_tensor = pad_to_square(img_tensor, img_size)
    
    # Финальная нормализация
    img_tensor = normalize_image(img_tensor, norm_range)
    
    return img_tensor


# -----------------------------
# Dataset для LQ → HQ
# -----------------------------

class FloorPlanDataset(Dataset):
    """
    Dataset для работы с парами изображений LQ → HQ из ClearML.
    
    Args:
        dataset_root: Путь к локальной копии датасета ClearML
        img_size: Размер стороны квадрата
        norm_range: Диапазон нормализации ("0_1" или "-1_1")
    """
    
    def __init__(self, dataset_root: Path, img_size: int = 512, norm_range: str = "0_1"):
        self.img_size = img_size
        self.norm_range = norm_range
        
        lq_dir = dataset_root / 'LQ'
        hq_dir = dataset_root / 'HQ'
        
        # Поддержка jpg и png
        valid_ext = ('.jpg', '.jpeg', '.png')
        
        self.lq_paths = sorted([
            lq_dir / f for f in os.listdir(lq_dir) 
            if f.lower().endswith(valid_ext)
        ])
        self.hq_paths = sorted([
            hq_dir / f for f in os.listdir(hq_dir) 
            if f.lower().endswith(valid_ext)
        ])
        
        if len(self.lq_paths) != len(self.hq_paths):
            raise RuntimeError(
                f"LQ и HQ имеют разное количество файлов: "
                f"{len(self.lq_paths)} vs {len(self.hq_paths)}"
            )
        
        if len(self.lq_paths) == 0:
            raise RuntimeError("Датасет пуст! Проверьте путь к датасету.")
    
    def __len__(self) -> int:
        return len(self.lq_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq_img = preprocess_image(self.lq_paths[idx], self.img_size, self.norm_range)
        hq_img = preprocess_image(self.hq_paths[idx], self.img_size, self.norm_range)
        return lq_img, hq_img


# -----------------------------
# Функции сохранения/загрузки состояния DataLoader
# -----------------------------

def save_dataloader_state(dataloader: DataLoader, filepath: str) -> None:
    """
    Сохраняет состояние DataLoader (состояние генератора для shuffle).
    
    Args:
        dataloader: DataLoader для сохранения
        filepath: Путь для сохранения состояния
    """
    if dataloader.generator is None:
        raise ValueError("DataLoader должен быть создан с shuffle=True для сохранения состояния")
    
    state = {
        'generator_state': dataloader.generator.get_state(),
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers,
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)


def load_dataloader_state(
    filepath: str, 
    dataset: Dataset
) -> DataLoader:
    """
    Загружает состояние DataLoader и создаёт новый DataLoader с тем же состоянием.
    
    Args:
        filepath: Путь к файлу с состоянием
        dataset: Dataset для использования в DataLoader
        
    Returns:
        DataLoader с восстановленным состоянием
    """
    with open(filepath, 'rb') as f:
        state = pickle.load(f)
    
    # Создаём генератор и восстанавливаем состояние
    generator = torch.Generator()
    generator.set_state(state['generator_state'])
    
    # Создаём DataLoader с восстановленным генератором
    dataloader = DataLoader(
        dataset, 
        batch_size=state['batch_size'], 
        shuffle=True, 
        generator=generator,
        num_workers=state['num_workers']
    )
    
    return dataloader


# -----------------------------
# Основная функция создания DataLoaders
# -----------------------------

def create_dataloaders(
    dataset_id: str,
    image_size: int = 512,
    norm_range: str = "0_1",
    batch_size: int = 16,
    train_split: float = 0.8,
    num_workers: int = 0,
    save_state: bool = False,
    state_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Создаёт train и test DataLoaders из ClearML датасета.
    
    Args:
        dataset_id: ID датасета ClearML из Task 1
        image_size: Размер стороны квадрата
        norm_range: Диапазон нормализации ("0_1" или "-1_1")
        batch_size: Размер батча
        train_split: Доля train выборки
        num_workers: Количество рабочих процессов для загрузки данных
        save_state: Сохранять ли состояние DataLoader
        state_dir: Директория для сохранения состояния (по умолчанию - текущая)
        
    Returns:
        train_loader, test_loader
    """
    # Загружаем датасет из ClearML
    print(f"Загрузка датасета ClearML: {dataset_id}")
    clearml_dataset = ClearMLDataset.get(dataset_id=dataset_id)
    
    # Получаем локальную копию
    local_path = Path(clearml_dataset.get_local_copy())
    print(f"Локальная копия датасета: {local_path}")
    
    # Создаём полный датасет
    full_dataset = FloorPlanDataset(
        local_path, 
        img_size=image_size, 
        norm_range=norm_range
    )
    print(f"Размер полного датасета: {len(full_dataset)} изображений")
    
    # Разбиваем на train/test
    train_size = int(train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Используем generator для воспроизводимости
    generator = torch.Generator()
    generator.manual_seed(42)
    
    train_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, test_size],
        generator=generator
    )
    
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Создаём DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0
    )
    
    # Сохраняем состояние если нужно
    if save_state:
        if state_dir is None:
            state_dir = "."
        
        Path(state_dir).mkdir(parents=True, exist_ok=True)
        
        train_state_path = os.path.join(state_dir, "train_dataloader_state.pkl")
        test_state_path = os.path.join(state_dir, "test_dataloader_state.pkl")
        
        # Для test не сохраняем состояние (shuffle=False)
        save_dataloader_state(train_loader, train_state_path)
        print(f"Состояние train DataLoader сохранено: {train_state_path}")
    
    return train_loader, test_loader


# -----------------------------
# CLI интерфейс для запуска через ClearML Agent
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Второй этап обработки датасета: создание PyTorch DataLoaders из ClearML"
    )
    
    # Обязательные параметры
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="ID датасета из Task 1 (pipeline передаёт автоматически)"
    )
    
    # Параметры предобработки
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Размер стороны квадрата (500, 1000, 2000 и т.д.). По умолчанию 512"
    )
    
    parser.add_argument(
        "--norm_range",
        type=str,
        default="0_1",
        choices=["0_1", "-1_1"],
        help="Диапазон нормализации: '0_1' для [0,1] или '-1_1' для [-1,1]. По умолчанию '0_1'"
    )
    
    # Параметры DataLoader
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Размер батча. По умолчанию 16"
    )
    
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Доля train выборки. По умолчанию 0.8"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Количество рабочих процессов для загрузки данных. По умолчанию 0"
    )
    
    # Параметры сохранения состояния
    parser.add_argument(
        "--save_state",
        action="store_true",
        help="Сохранить состояние DataLoader для воспроизводимости"
    )
    
    parser.add_argument(
        "--state_dir",
        type=str,
        default=".",
        help="Директория для сохранения состояния DataLoader"
    )
    
    # Параметры ClearML Task
    parser.add_argument(
        "--project_name",
        type=str,
        default="Image_Restoration",
        help="Имя проекта ClearML"
    )
    
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="Имя задачи ClearML (по умолчанию генерируется автоматически)"
    )
    
    parser.add_argument(
        "--execute_remotely",
        action="store_true",
        help="Выполнить задачу удалённо через ClearML Agent"
    )
    
    parser.add_argument(
        "--queue",
        type=str,
        default="default",
        help="Очередь для удалённого выполнения"
    )
    
    return parser.parse_args()



args = parse_args()

# Проверяем наличие CUDA для правильной установки torch (только для удалённого выполнения)
# ВАЖНО: ignore_requirements/add_requirements должны вызываться ДО Task.init()
if args.execute_remotely:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"\n{'=' * 60}")
    print("Удалённое выполнение через ClearML Agent")
    print(f"{'=' * 60}")
    print(f"CUDA доступен: {cuda_available}")

    # Если CUDA нет - игнорируем platform-specific зависимости из uv.lock
    # и устанавливаем CPU-версию torch
    if not cuda_available:
        print("CUDA не доступен, используем CPU-версию torch")
        Task.ignore_requirements("torch")
        Task.ignore_requirements("torchvision")
        Task.ignore_requirements("torchaudio")
        Task.add_requirements("torch", ">=2.0.0")
        Task.add_requirements("torchvision", ">=0.15.0")
    else:
        print("CUDA доступен, используем GPU-версию torch")
    print(f"{'=' * 60}\n")

# Инициализация Task (единая для локального и удалённого выполнения)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
task_name = args.task_name or f"pytorch_dataset_{timestamp}"

task = Task.init(
    project_name=args.project_name,
    task_name=task_name,
    task_type=Task.TaskTypes.data_processing,
    reuse_last_task_id=False,
)

# Подключение конфигурации
config = {
    "dataset_id": args.dataset_id,
    "image_size": args.image_size,
    "norm_range": args.norm_range,
    "batch_size": args.batch_size,
    "train_split": args.train_split,
    "num_workers": args.num_workers,
    "save_state": args.save_state,
    "state_dir": args.state_dir,
}
task.connect_configuration(config)
# Для удалённого выполнения - отправляем задачу в очередь сразу после init
if args.execute_remotely:
    # Отправляем задачу на выполнение в очередь
    print(f"Отправка задачи в очередь '{args.queue}'...")

    task.execute_remotely(
        queue_name=args.queue,
        clone=False,
        exit_process=True,
    )

# Локальное выполнение (без --execute_remotely)

# Создание DataLoaders
print("\n" + "=" * 60)
print("Создание DataLoaders")
print("=" * 60)
print(f"Dataset ID: {args.dataset_id}")
print(f"Image Size: {args.image_size}")
print(f"Norm Range: {args.norm_range}")
print(f"Batch Size: {args.batch_size}")
print(f"Train Split: {args.train_split}")
print("=" * 60 + "\n")

train_loader, test_loader = create_dataloaders(
    dataset_id=args.dataset_id,
    image_size=args.image_size,
    norm_range=args.norm_range,
    batch_size=args.batch_size,
    train_split=args.train_split,
    num_workers=args.num_workers,
    save_state=args.save_state,
    state_dir=args.state_dir,
)

# Проверка
print("\nПроверка DataLoaders:")
lq, hq = next(iter(train_loader))
print(f"  Train batch LQ: {lq.shape}")
print(f"  Train batch HQ: {hq.shape}")

lq, hq = next(iter(test_loader))
print(f"  Test batch LQ: {lq.shape}")
print(f"  Test batch HQ: {hq.shape}")

# Логирование статистики
task.get_logger().report_scalar(
    title="Dataset Stats",
    series="train_size",
    value=len(train_loader.dataset),
    iteration=0
)
task.get_logger().report_scalar(
    title="Dataset Stats",
    series="test_size",
    value=len(test_loader.dataset),
    iteration=0
)

# Логирование примеров изображений для отладки (2-3 sample)
print("\nЛогирование примеров изображений в ClearML...")
logger = task.get_logger()

# Берём 3 примера из train
num_samples = min(3, len(train_loader.dataset))
for idx in range(num_samples):
    lq_sample, hq_sample = train_loader.dataset[idx]
    
    # Конвертация (C, H, W) -> (H, W, C) для отображения
    lq_np = lq_sample.permute(1, 2, 0).cpu().numpy()
    hq_np = hq_sample.permute(1, 2, 0).cpu().numpy()
    
    # Если нормализация в [-1, 1], конвертируем обратно в [0, 1] для отображения
    if args.norm_range == "-1_1":
        lq_np = (lq_np + 1.0) / 2.0
        hq_np = (hq_np + 1.0) / 2.0
    
    # Ограничиваем значения в [0, 1]
    lq_np = np.clip(lq_np, 0, 1)
    hq_np = np.clip(hq_np, 0, 1)
    
    # Логирование LQ
    logger.report_image(
        title="Debug Samples - LQ (Low Quality)",
        series=f"lq_sample_{idx}",
        iteration=0,
        image=lq_np
    )

    # Логирование HQ
    logger.report_image(
        title="Debug Samples - HQ (High Quality)",
        series=f"hq_sample_{idx}",
        iteration=0,
        image=hq_np
    )

print(f"  Загружено {num_samples} примеров в ClearML logger")

# Сохранение DataLoaders как артефактов ClearML
print("\nСохранение DataLoaders как артефактов ClearML...")
import tempfile
import shutil

# Создаём временную директорию для сохранения состояний
with tempfile.TemporaryDirectory() as temp_dir:
    temp_path = Path(temp_dir)
    
    # Сохраняем состояния DataLoaders
    train_state_path = temp_path / "train_dataloader_state.pkl"
    test_state_path = temp_path / "test_dataloader_state.pkl"
    
    # Сохраняем состояние train DataLoader
    if train_loader.generator is not None:
        save_dataloader_state(train_loader, str(train_state_path))
    else:
        # Если generator нет, сохраняем метаданные
        train_metadata = {
            'batch_size': train_loader.batch_size,
            'num_workers': train_loader.num_workers,
            'dataset_size': len(train_loader.dataset),
            'num_batches': len(train_loader),
            'shuffle': True
        }
        with open(train_state_path, 'wb') as f:
            pickle.dump(train_metadata, f)
    print(f"  Состояние train DataLoader сохранено")

    # Сохраняем состояние test DataLoader
    test_metadata = {
        'batch_size': test_loader.batch_size,
        'num_workers': test_loader.num_workers,
        'dataset_size': len(test_loader.dataset),
        'num_batches': len(test_loader),
        'shuffle': False
    }
    with open(test_state_path, 'wb') as f:
        pickle.dump(test_metadata, f)
    print(f"  Состояние test DataLoader сохранено")
    
    # Сохраняем метаданные датасета
    dataset_metadata = {
        'dataset_id': args.dataset_id,
        'image_size': args.image_size,
        'norm_range': args.norm_range,
        'batch_size': args.batch_size,
        'train_split': args.train_split,
        'train_size': len(train_loader.dataset),
        'test_size': len(test_loader.dataset),
    }
    metadata_path = temp_path / "dataset_metadata.json"
    import json
    with open(metadata_path, 'w') as f:
        json.dump(dataset_metadata, f, indent=2)
    print(f"  Метаданные датасета сохранены")
    
    # Загружаем артефакты в ClearML
    task.upload_artifact(
        name='dataloader_states',
        artifact_object=str(temp_path),
        delete_after_upload=False
    )

print(f"  DataLoaders сохранены как артефакт 'dataloader_states'")

print("\n" + "=" * 60)
print("ГОТОВО")
print("=" * 60)
print(f"Train DataLoader: {len(train_loader)} батчей")
print(f"Test DataLoader: {len(test_loader)} батчей")

if args.save_state:
    print(f"Состояние сохранено в: {args.state_dir}")

task.close()

