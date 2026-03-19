"""
Создание ClearML Dataset с парами изображений HQ/LQ.

LQ создаётся из HQ путём:
- Случайное снижение JPEG качества (jpeg_quality_min .. jpeg_quality_max)
- Случайное снижение разрешения (resolution_ratio_min .. resolution_ratio_max) с последующим растягиванием

Без добавления шума!

Источники данных:
- Архив train-00.tar.xz
- Локальная папка с изображениями
- ClearML Dataset по ID
"""

import tarfile
import lzma
import cv2
import numpy as np
from pathlib import Path
import random
import argparse
import shutil
from datetime import datetime
from clearml import Task, Dataset


def extract_coco_vis_pngs(archive_path: Path, output_dir: Path) -> list[Path]:
    """
    Потоковое извлечение PNG файлов из coco_vis папки архива.
    Не загружает всё в память одновременно.
    Возвращает список путей к извлечённым файлам.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted_files = []

    print(f"Распаковка {archive_path}...")
    with lzma.open(archive_path, 'rb') as xz:
        with tarfile.open(fileobj=xz, mode='r|') as tar:
            for member in tar:
                # Извлекаем только файлы из coco_vis папки
                if 'coco_vis' in member.name and member.name.endswith('.png'):
                    # Сохраняем только имя файла, без структуры папок
                    filename = Path(member.name).name
                    output_path = output_dir / filename

                    f = tar.extractfile(member)
                    if f:
                        with open(output_path, 'wb') as out:
                            out.write(f.read())
                        extracted_files.append(output_path)

                        if len(extracted_files) % 500 == 0:
                            print(f"  Извлечено: {len(extracted_files)} PNG")

    print(f"Всего извлечено: {len(extracted_files)} PNG из coco_vis")
    return extracted_files


def load_from_local_folder(folder_path: Path) -> list[Path]:
    """
    Загрузка списка изображений из локальной папки.
    Поддерживаются форматы: png, jpg, jpeg, bmp.
    """
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = set()
    for ext in extensions:
        for f in folder_path.glob(ext):
            files.add(f)
        for f in folder_path.glob(ext.upper()):
            files.add(f)
    
    files = sorted(files)
    print(f"Найдено {len(files)} изображений в {folder_path}")
    return files


def download_from_clearml(dataset_id: str, output_dir: Path) -> list[Path]:
    """
    Загрузка изображений из ClearML Dataset по ID.
    Скачивает папку coco_vis (исходные изображения).
    """
    print(f"Загрузка ClearML Dataset {dataset_id}...")

    dataset = Dataset.get(dataset_id=dataset_id)

    # Получаем локальную копию датасета
    local_path = dataset.get_local_copy()

    # Ищем папку coco_vis внутри локальной копии
    coco_vis_source = Path(local_path) / "coco_vis"
    
    if not coco_vis_source.exists():
        # Пробуем найти coco_vis в подпапках
        coco_vis_source = None
        for folder in Path(local_path).rglob("coco_vis"):
            if folder.is_dir():
                coco_vis_source = folder
                break
        
        if coco_vis_source is None:
            # Если не нашли, используем корень
            coco_vis_source = Path(local_path)
            print(f"  Папка coco_vis не найдена, используем: {coco_vis_source}")
        else:
            print(f"  Найдена папка coco_vis: {coco_vis_source}")
    
    # Копируем файлы в рабочую директорию
    download_dir = output_dir / "clearml_download" / "coco_vis"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    # Копируем PNG файлы из coco_vis
    for file in coco_vis_source.glob("*.png"):
        if file.is_file():
            shutil.copy2(str(file), str(download_dir / file.name))

    # Получаем список файлов
    files = load_from_local_folder(download_dir)
    print(f"Загружено {len(files)} изображений из ClearML")

    return files


def create_lq_image(hq_image: np.ndarray, jpeg_quality_min: int, jpeg_quality_max: int,
                    resolution_ratio_min: int, resolution_ratio_max: int) -> np.ndarray:
    """
    Создаёт LQ версию изображения путём:
    1. Случайное снижение разрешения (downscale + upscale)
    2. Случайное JPEG сжатие
    
    Без добавления шума!
    """
    h, w = hq_image.shape[:2]
    
    # 1. Снижение разрешения
    ratio = random.randint(resolution_ratio_min, resolution_ratio_max)
    small_w, small_h = w // ratio, h // ratio
    small = cv2.resize(hq_image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    img_upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # 2. JPEG сжатие
    jpeg_quality = random.randint(jpeg_quality_min, jpeg_quality_max)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, jpeg_data = cv2.imencode('.jpg', img_upscaled, encode_param)
    img_lq = cv2.imdecode(jpeg_data, cv2.IMREAD_COLOR)
    
    return img_lq


def process_image(hq_path: Path, hq_dir: Path, lq_dir: Path, idx: int,
                  jpeg_quality_min: int, jpeg_quality_max: int,
                  resolution_ratio_min: int, resolution_ratio_max: int) -> bool:
    """
    Обработка одного изображения:
    - Сохранение HQ как PNG
    - Создание LQ и сохранение как JPG
    """
    try:
        # Чтение HQ
        hq_image = cv2.imread(str(hq_path), cv2.IMREAD_COLOR)
        if hq_image is None:
            return False

        # Сохранение HQ как PNG
        hq_dest = hq_dir / f"{idx:06d}.png"
        cv2.imwrite(str(hq_dest), hq_image)

        # Создание LQ
        lq_image = create_lq_image(
            hq_image, jpeg_quality_min, jpeg_quality_max,
            resolution_ratio_min, resolution_ratio_max
        )
        # Сохранение LQ как JPG (меньший размер)
        lq_dest = lq_dir / f"{idx:06d}.jpg"
        cv2.imwrite(str(lq_dest), lq_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        return True
    except Exception as e:
        print(f"  Ошибка при обработке {hq_path}: {e}")
        return False


def create_clearml_dataset(hq_lq_dir: Path, dataset_name: str,
                           jpeg_quality_min: int, jpeg_quality_max: int,
                           resolution_ratio_min: int, resolution_ratio_max: int) -> str:
    """
    Создание ClearML Dataset из папок HQ/LQ.
    """
    print("\nСоздание ClearML Dataset...")
    
    dataset = Dataset.create(
        dataset_project="Image_Restoration",
        dataset_name=dataset_name,
        description=(
            f"Пары HQ/LQ изображений.\n"
            f"LQ: JPEG {jpeg_quality_min}-{jpeg_quality_max}, "
            f"downscale 1/{resolution_ratio_min}-1/{resolution_ratio_max}"
        )
    )
    
    # Добавляем папки в датасет
    print("Добавление HQ папки...")
    dataset.add_files(
        path=str(hq_lq_dir / "HQ"),
        dataset_path="HQ",
        verbose=False
    )
    
    print("Добавление LQ папки...")
    dataset.add_files(
        path=str(hq_lq_dir / "LQ"),
        dataset_path="LQ",
        verbose=False
    )
    
    print("Загрузка на сервер ClearML...")
    dataset.upload(show_progress=True)
    dataset.finalize()
    
    print(f"\nDataset ID: {dataset.id}")
    print(f"Dataset Name: {dataset.name}")
    
    return dataset.id


def main():
    # Подключение параметров через args
    parser = argparse.ArgumentParser()
    
    # Параметры деградации
    parser.add_argument(
        "--jpeg_quality_min",
        type=int,
        default=10,
        help="Минимальное качество JPEG для LQ"
    )
    parser.add_argument(
        "--jpeg_quality_max",
        type=int,
        default=50,
        help="Максимальное качество JPEG для LQ"
    )
    parser.add_argument(
        "--resolution_ratio_min",
        type=int,
        default=2,
        help="Минимальный коэффициент снижения разрешения"
    )
    parser.add_argument(
        "--resolution_ratio_max",
        type=int,
        default=4,
        help="Максимальный коэффициент снижения разрешения"
    )
    
    # Параметры вывода
    parser.add_argument(
        "--output_dir",
        type=str,
        default="floorplan_dataset",
        help="Папка для выходных данных"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="FloorPlanCAD_HQ_LQ",
        help="Имя датасета ClearML"
    )
    
    # Источник данных (взаимоисключающие)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--archive_path",
        type=str,
        help="Путь к архиву .tar.xz с исходными изображениями"
    )
    source_group.add_argument(
        "--local_folder",
        type=str,
        help="Путь к локальной папке с изображениями"
    )
    source_group.add_argument(
        "--clearml_dataset_id",
        type=str,
        help="ID ClearML Dataset для загрузки исходных изображений"
    )

    args = parser.parse_args()

    # Инициализация ClearML Task с уникальным именем
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task = Task.init(
        project_name='Image_Restoration',
        task_name=f'create_floorplan_dataset_{timestamp}',
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,  # Не перезаписывать предыдущую задачу
    )

    # Подключение параметров к Task
    config = {
        "jpeg_quality_min": args.jpeg_quality_min,
        "jpeg_quality_max": args.jpeg_quality_max,
        "resolution_ratio_min": args.resolution_ratio_min,
        "resolution_ratio_max": args.resolution_ratio_max,
        "output_dir": args.output_dir,
        "dataset_name": args.dataset_name,
    }
    if args.archive_path:
        config["archive_path"] = args.archive_path
        config["source_type"] = "archive"
    elif args.local_folder:
        config["local_folder"] = args.local_folder
        config["source_type"] = "local_folder"
    else:
        config["clearml_dataset_id"] = args.clearml_dataset_id
        config["source_type"] = "clearml"
    
    task.connect_configuration(config)

    output_dir = Path(args.output_dir)
    temp_dir = output_dir / "raw_pngs"
    hq_dir = output_dir / "HQ"
    lq_dir = output_dir / "LQ"

    # Создание папок
    hq_dir.mkdir(parents=True, exist_ok=True)
    lq_dir.mkdir(parents=True, exist_ok=True)

    # Шаг 1: Загрузка исходных изображений
    if args.archive_path:
        archive_path = Path(args.archive_path)
        if archive_path.exists():
            raw_files = extract_coco_vis_pngs(archive_path, temp_dir)
        else:
            print(f"Архив не найден: {archive_path}")
            return
    elif args.local_folder:
        folder_path = Path(args.local_folder)
        raw_files = load_from_local_folder(folder_path)
    else:  # clearml_dataset_id
        raw_files = download_from_clearml(args.clearml_dataset_id, output_dir)

    if not raw_files:
        print("Нет файлов для обработки!")
        return

    # Шаг 2: Обработка изображений (потоково, по одному)
    print(f"\nОбработка {len(raw_files)} изображений...")
    print(f"Параметры: JPEG={args.jpeg_quality_min}-{args.jpeg_quality_max}, "
          f"Downscale=1/{args.resolution_ratio_min}-1/{args.resolution_ratio_max}")

    success_count = 0
    for idx, hq_path in enumerate(sorted(raw_files)):
        ok = process_image(
            hq_path, hq_dir, lq_dir, idx,
            args.jpeg_quality_min, args.jpeg_quality_max,
            args.resolution_ratio_min, args.resolution_ratio_max
        )
        if ok:
            success_count += 1

        if (idx + 1) % 100 == 0:
            print(f"  Обработано: {idx + 1}/{len(raw_files)}")

    print(f"\nОбработано успешно: {success_count}/{len(raw_files)}")

    # Шаг 3: Создание ClearML Dataset
    dataset_id = create_clearml_dataset(
        output_dir,
        args.dataset_name,
        args.jpeg_quality_min,
        args.jpeg_quality_max,
        args.resolution_ratio_min,
        args.resolution_ratio_max
    )
    print(f"\nDataset создан: {dataset_id}")

    # Сохраняем dataset_id в task parameters
    task.connect_configuration({"dataset_id": dataset_id})
    
    # Завершаем задачу
    task.close()


if __name__ == "__main__":
    main()
