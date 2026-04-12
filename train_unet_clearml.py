"""
Обучение U-Net для восстановления схем с интеграцией в ClearML.

Скрипт:
- принимает ID задачи preprocessing этапа в ClearML;
- скачивает артефакт `dataloader_states` и читает `dataset_metadata.json`;
- заново создаёт PyTorch Dataset/DataLoader на основе исходного ClearML Dataset;
- обучает U-Net локально или удалённо через ClearML Agent;
- логирует метрики, debug samples и загружает модель как артефакт в ClearML.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from clearml import Dataset as ClearMLDataset
from clearml import Task
from torch.utils.data import DataLoader, Dataset, Subset, random_split

import sys

if sys.platform == "win32":
    Task.ignore_requirements("pywin32")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pad_to_square(image: torch.Tensor, target_size: int) -> torch.Tensor:
    c, h, w = image.shape
    if h == w == target_size:
        return image

    pad_h = target_size - h
    pad_w = target_size - w
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
    return torch.nn.functional.pad(image, padding, mode="constant", value=0.0)


def normalize_image(image: torch.Tensor, norm_range: str = "0_1") -> torch.Tensor:
    if norm_range == "0_1":
        return image
    if norm_range == "-1_1":
        return image * 2.0 - 1.0
    raise ValueError(f"Unsupported norm_range: {norm_range}")


def denormalize_image(image: torch.Tensor, norm_range: str = "0_1") -> torch.Tensor:
    if norm_range == "0_1":
        return image
    if norm_range == "-1_1":
        return (image + 1.0) / 2.0
    raise ValueError(f"Unsupported norm_range: {norm_range}")


def preprocess_image(path: Path, img_size: int, norm_range: str = "0_1") -> torch.Tensor:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = min(img_size / w, img_size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0

    img_tensor = torch.from_numpy(img).permute(2, 0, 1)
    img_tensor = pad_to_square(img_tensor, img_size)
    img_tensor = normalize_image(img_tensor, norm_range)
    return img_tensor


class FloorPlanDataset(Dataset):
    def __init__(self, dataset_root: Path, img_size: int = 512, norm_range: str = "0_1"):
        self.img_size = img_size
        self.norm_range = norm_range

        lq_dir = dataset_root / "LQ"
        hq_dir = dataset_root / "HQ"
        valid_ext = (".jpg", ".jpeg", ".png")

        self.lq_paths = sorted([lq_dir / f for f in os.listdir(lq_dir) if f.lower().endswith(valid_ext)])
        self.hq_paths = sorted([hq_dir / f for f in os.listdir(hq_dir) if f.lower().endswith(valid_ext)])

        if len(self.lq_paths) != len(self.hq_paths):
            raise RuntimeError(f"LQ/HQ file count mismatch: {len(self.lq_paths)} vs {len(self.hq_paths)}")
        if not self.lq_paths:
            raise RuntimeError("Dataset is empty")

    def __len__(self) -> int:
        return len(self.lq_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lq_img = preprocess_image(self.lq_paths[idx], self.img_size, self.norm_range)
        hq_img = preprocess_image(self.hq_paths[idx], self.img_size, self.norm_range)
        return lq_img, hq_img


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        depth: int = 4,
        max_channels: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("UNet depth must be at least 2")

        features = [min(base_channels * (2 ** i), max_channels) for i in range(depth)]

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        current_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(current_in, feature, dropout=dropout))
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            current_in = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout=dropout)

        self.up_transpose = nn.ModuleList()
        self.ups = nn.ModuleList()
        current_in = features[-1] * 2
        for feature in reversed(features):
            self.up_transpose.append(nn.ConvTranspose2d(current_in, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature * 2, feature, dropout=dropout))
            current_in = feature

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx](x)

        x = self.final_conv(x)
        return torch.sigmoid(x)


@dataclass
class TrainConfig:
    preprocess_task_id: str
    dataloader_artifact_name: str = "dataloader_states"
    project_name: str = "Image_Restoration"
    task_name: Optional[str] = None
    execute_remotely: bool = False
    queue: str = "default"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 20
    batch_size: Optional[int] = None
    num_workers: int = 0
    train_fraction: float = 1.0
    val_fraction: float = 1.0
    loss: str = "l1"
    scheduler: str = "none"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    base_channels: int = 32
    depth: int = 4
    max_channels: int = 256
    dropout: float = 0.0
    seed: int = 42
    device: str = "auto"
    debug_samples: int = 3
    save_every_epoch: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train U-Net for circuit restoration with ClearML integration")

    parser.add_argument("--preprocess_task_id", type=str, required=True, help="ClearML Task ID of preprocessing step")
    parser.add_argument("--dataloader_artifact_name", type=str, default="dataloader_states", help="Artifact name from preprocessing task")

    parser.add_argument("--project_name", type=str, default="Image_Restoration", help="ClearML project name")
    parser.add_argument("--task_name", type=str, default=None, help="ClearML task name")
    parser.add_argument("--execute_remotely", action="store_true", help="Queue task to ClearML Agent")
    parser.add_argument("--queue", type=str, default="default", help="ClearML queue name")

    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw", "sgd"], help="Optimizer")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override preprocessing batch size")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--train_fraction", type=float, default=1.0, help="Fraction of train subset to use, from 0 to 1")
    parser.add_argument("--val_fraction", type=float, default=1.0, help="Fraction of val/test subset to use, from 0 to 1")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse", "smooth_l1"], help="Loss function")
    parser.add_argument("--scheduler", type=str, default="none", choices=["none", "step"], help="Scheduler type")
    parser.add_argument("--scheduler_step_size", type=int, default=10, help="StepLR step size")
    parser.add_argument("--scheduler_gamma", type=float, default=0.5, help="StepLR gamma")
    parser.add_argument("--base_channels", type=int, default=32, help="Base number of U-Net channels")
    parser.add_argument("--depth", type=int, default=4, help="Number of encoder/decoder stages in U-Net")
    parser.add_argument("--max_channels", type=int, default=256, help="Upper cap for channels in deeper U-Net blocks")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout in convolution blocks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Execution device")
    parser.add_argument("--debug_samples", type=int, default=3, help="Number of debug samples to log")
    parser.add_argument("--save_every_epoch", action="store_true", help="Upload per-epoch checkpoints")

    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_remote_environment(execute_remotely: bool, device_arg: str) -> None:
    if not execute_remotely:
        return

    cuda_available = torch.cuda.is_available()
    expects_remote_cuda = device_arg == "cuda"
    print("\n" + "=" * 60)
    print("Remote execution via ClearML Agent")
    print("=" * 60)
    print(f"Local CUDA available: {cuda_available}")
    print(f"Requested device: {device_arg}")

    # Важный случай: локальная машина может быть без CUDA, но агент в очереди - с GPU.
    # Если пользователь явно просит --device cuda, не подменяем зависимости на CPU.
    if not cuda_available and not expects_remote_cuda:
        Task.ignore_requirements("torch")
        Task.ignore_requirements("torchvision")
        Task.ignore_requirements("torchaudio")
        Task.add_requirements("torch", ">=2.0.0")
        Task.add_requirements("torchvision", ">=0.15.0")
        print("Configured CPU torch requirements for agent")
    elif expects_remote_cuda:
        print("Keeping GPU-oriented torch requirements for the remote agent")
    print("=" * 60 + "\n")


def init_task(config: TrainConfig) -> Task:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = config.task_name or f"train_unet_{timestamp}"
    task = Task.init(
        project_name=config.project_name,
        task_name=task_name,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )
    task.connect(asdict(config))
    return task


def download_preprocessing_metadata(preprocess_task_id: str, artifact_name: str) -> Dict:
    preprocess_task = Task.get_task(task_id=preprocess_task_id)
    if artifact_name not in preprocess_task.artifacts:
        available = ", ".join(sorted(preprocess_task.artifacts.keys()))
        raise KeyError(f"Artifact '{artifact_name}' not found. Available: {available}")

    artifact = preprocess_task.artifacts[artifact_name]
    local_copy = Path(artifact.get_local_copy(extract_archive=True))

    if local_copy.is_file():
        extracted_dir = Path(tempfile.mkdtemp(prefix="clearml_dataloader_artifact_"))
        shutil.unpack_archive(str(local_copy), str(extracted_dir))
        artifact_dir = extracted_dir
    else:
        artifact_dir = local_copy

    metadata_path = artifact_dir / "dataset_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset_metadata.json not found in artifact: {artifact_dir}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_datasets(
    dataset_id: str,
    image_size: int,
    norm_range: str,
    train_split: float,
    seed: int,
    train_fraction: float,
    val_fraction: float,
) -> Tuple[Dataset, Dataset, Path]:
    clearml_dataset = ClearMLDataset.get(dataset_id=dataset_id)
    dataset_root = Path(clearml_dataset.get_local_copy())
    full_dataset = FloorPlanDataset(dataset_root, img_size=image_size, norm_range=norm_range)

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = apply_fraction(train_dataset, train_fraction, seed)
    val_dataset = apply_fraction(val_dataset, val_fraction, seed + 1)
    return train_dataset, val_dataset, dataset_root


def apply_fraction(dataset: Dataset, fraction: float, seed: int) -> Dataset:
    if not (0 < fraction <= 1.0):
        raise ValueError("Fraction must be in (0, 1]")
    if fraction >= 1.0:
        return dataset

    target_len = max(1, int(len(dataset) * fraction))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:target_len].tolist()
    return Subset(dataset, indices)


def build_dataloaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, num_workers: int, seed: int) -> Tuple[DataLoader, DataLoader]:
    train_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def create_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def create_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig):
    if config.scheduler == "none":
        return None
    if config.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.scheduler_step_size, gamma=config.scheduler_gamma)
    raise ValueError(f"Unsupported scheduler: {config.scheduler}")


def create_loss(loss_name: str) -> nn.Module:
    if loss_name == "l1":
        return nn.L1Loss()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "smooth_l1":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_psnr = 0.0
    total_items = 0

    for lq, hq in loader:
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            pred = model(lq)
            loss = criterion(pred, hq)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = lq.size(0)
        total_items += batch_size
        total_loss += loss.item() * batch_size
        total_psnr += compute_psnr(pred.detach(), hq.detach()) * batch_size

    return total_loss / total_items, total_psnr / total_items


def tensor_to_image(tensor: torch.Tensor, norm_range: str) -> np.ndarray:
    tensor = denormalize_image(tensor.detach().cpu(), norm_range=norm_range)
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def log_debug_samples(
    task: Task,
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    norm_range: str,
    iteration: int,
    num_samples: int,
) -> None:
    logger = task.get_logger()
    model.eval()

    count = min(num_samples, len(dataset))
    if count == 0:
        return

    with torch.no_grad():
        for idx in range(count):
            lq, hq = dataset[idx]
            pred = model(lq.unsqueeze(0).to(device)).squeeze(0)

            logger.report_image(
                title="Debug Samples - Input",
                series=f"sample_{idx}",
                iteration=iteration,
                image=tensor_to_image(lq, norm_range),
            )
            logger.report_image(
                title="Debug Samples - Target",
                series=f"sample_{idx}",
                iteration=iteration,
                image=tensor_to_image(hq, norm_range),
            )
            logger.report_image(
                title="Debug Samples - Prediction",
                series=f"sample_{idx}",
                iteration=iteration,
                image=tensor_to_image(pred, norm_range),
            )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    metadata: Dict,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_loss": best_val_loss,
        "metadata": metadata,
    }
    torch.save(checkpoint, path)


def cleanup_directory(path: Path, retries: int = 5, delay_seconds: float = 1.0) -> None:
    if not path.exists():
        return

    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except PermissionError:
            if attempt == retries - 1:
                print(f"Warning: could not remove temporary directory: {path}")
                return
            time.sleep(delay_seconds)


def main() -> None:
    args = parse_args()
    config = TrainConfig(**vars(args))

    prepare_remote_environment(config.execute_remotely, config.device)
    task = init_task(config)

    if config.execute_remotely:
        print(f"Sending task to ClearML queue '{config.queue}'...")
        task.execute_remotely(queue_name=config.queue, clone=False, exit_process=True)

    seed_everything(config.seed)
    device = resolve_device(config.device)

    print("\n" + "=" * 60)
    print("Loading preprocessing metadata")
    print("=" * 60)
    metadata = download_preprocessing_metadata(config.preprocess_task_id, config.dataloader_artifact_name)
    dataset_id = metadata["dataset_id"]
    image_size = int(metadata["image_size"])
    norm_range = metadata["norm_range"]
    train_split = float(metadata["train_split"])
    batch_size = config.batch_size or int(metadata["batch_size"])

    print(f"Preprocess task ID: {config.preprocess_task_id}")
    print(f"Dataset ID: {dataset_id}")
    print(f"Image size: {image_size}")
    print(f"Norm range: {norm_range}")
    print(f"Train split: {train_split}")
    print(f"Batch size: {batch_size}")
    print(f"Train fraction: {config.train_fraction}")
    print(f"Val fraction: {config.val_fraction}")
    print(f"Device: {device}")

    train_dataset, val_dataset, dataset_root = build_datasets(
        dataset_id=dataset_id,
        image_size=image_size,
        norm_range=norm_range,
        train_split=train_split,
        seed=config.seed,
        train_fraction=config.train_fraction,
        val_fraction=config.val_fraction,
    )
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    task.get_logger().report_scalar("Dataset", "train_size", len(train_dataset), 0)
    task.get_logger().report_scalar("Dataset", "val_size", len(val_dataset), 0)

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=config.base_channels,
        depth=config.depth,
        max_channels=config.max_channels,
        dropout=config.dropout,
    ).to(device)
    criterion = create_loss(config.loss)
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)

    best_val_loss = float("inf")

    temp_path = Path(tempfile.mkdtemp(prefix="train_unet_clearml_"))
    try:
        best_model_path = temp_path / "best_model.pt"
        last_checkpoint_path = temp_path / "last_checkpoint.pt"

        training_metadata = {
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "image_size": image_size,
            "norm_range": norm_range,
            "train_split": train_split,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
            "batch_size": batch_size,
            "config": asdict(config),
        }

        print("\n" + "=" * 60)
        print("Training")
        print("=" * 60)

        for epoch in range(1, config.epochs + 1):
            train_loss, train_psnr = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_psnr = run_epoch(model, val_loader, criterion, None, device)

            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]

            task.get_logger().report_scalar("Loss", "train", train_loss, epoch)
            task.get_logger().report_scalar("Loss", "val", val_loss, epoch)
            task.get_logger().report_scalar("PSNR", "train", train_psnr, epoch)
            task.get_logger().report_scalar("PSNR", "val", val_psnr, epoch)
            task.get_logger().report_scalar("LR", "learning_rate", current_lr, epoch)

            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
                f"train_psnr={train_psnr:.3f} val_psnr={val_psnr:.3f} lr={current_lr:.6g}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    best_model_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    training_metadata,
                )
                task.upload_artifact(
                    name="best_model",
                    artifact_object=str(best_model_path),
                    delete_after_upload=False,
                )
                log_debug_samples(task, model, val_dataset, device, norm_range, epoch, config.debug_samples)

            if config.save_every_epoch:
                epoch_ckpt_path = temp_path / f"checkpoint_epoch_{epoch:03d}.pt"
                save_checkpoint(
                    epoch_ckpt_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_val_loss,
                    training_metadata,
                )
                task.upload_artifact(
                    name=f"checkpoint_epoch_{epoch:03d}",
                    artifact_object=str(epoch_ckpt_path),
                    delete_after_upload=False,
                )

        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            config.epochs,
            best_val_loss,
            training_metadata,
        )

        summary_path = temp_path / "training_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_loss": best_val_loss,
                    "epochs": config.epochs,
                    "dataset_id": dataset_id,
                    "train_size": len(train_dataset),
                    "val_size": len(val_dataset),
                    "device": str(device),
                    "config": asdict(config),
                    "preprocess_metadata": metadata,
                },
                f,
                indent=2,
            )

        task.upload_artifact("last_checkpoint", artifact_object=str(last_checkpoint_path), delete_after_upload=False)
        task.upload_artifact("training_summary", artifact_object=str(summary_path), delete_after_upload=False)
    finally:
        task.close()
        cleanup_directory(temp_path)

    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
