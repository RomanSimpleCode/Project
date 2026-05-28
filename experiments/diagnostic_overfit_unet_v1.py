#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install clearml pytorch-msssim')


# In[2]:


#!/usr/bin/env python
# coding: utf-8

"""
Diagnostic overfit-test for FloorPlanCAD LQ -> HQ restoration.

Goal:
- Take a tiny subset: 1-5 paired images.
- Train the model until it almost memorizes them.
- If losses do not drop strongly, the model/loss/data pipeline is wrong.

Designed for Google Colab + ClearML Dataset.
"""

import os
import random
import json
from pathlib import Path
from getpass import getpass

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from clearml import Task, Dataset as ClearMLDataset


# =========================================================
# CLEARML CREDENTIALS
# =========================================================
def setup_clearml_credentials():
    api_host = os.environ.get("CLEARML_API_HOST", "https://api.clear.ml")
    web_host = os.environ.get("CLEARML_WEB_HOST", "https://app.clear.ml")
    files_host = os.environ.get("CLEARML_FILES_HOST", "https://files.clear.ml")
    key = os.environ.get("ZCGVG8PSQJOOZNASXTHNO3MLHWXK6G")
    secret = os.environ.get("KQ46iRWMS_IRDUuB8BggNUZwj-3e_0CvEWmQXJU3pIWfanybc6smB4tfxxGKbQ1qSKI")

    if not key:
        key = getpass("ClearML API access key: ")
        os.environ["CLEARML_API_ACCESS_KEY"] = key

    if not secret:
        secret = getpass("ClearML API secret key: ")
        os.environ["CLEARML_API_SECRET_KEY"] = secret

    Task.set_credentials(
        api_host=api_host,
        web_host=web_host,
        files_host=files_host,
        key=key,
        secret=secret,
    )


# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    "project_name": "Vosstanovlenie_tehnicheskih_sistem",
    "task_name": "diagnostic_overfit_unet_v1",
    "clearml_dataset_id": "aa90ed3c5ca14bec9f9828a8891a5a59",

    "image_size": 1024,
    "patch_size": 512,
    "num_images": 3,
    "patches_per_image": 16,
    "batch_size": 2,
    "num_workers": 2,

    "epochs": 300,
    "lr": 2e-4,
    "weight_decay": 0.0,
    "base_channels": 48,

    # Much larger than 0.02. For diagnostic, the model must be allowed to change pixels.
    "residual_scale": 0.25,

    "white_threshold": 245,
    "min_content_ratio": 0.01,
    "max_crop_attempts": 100,

    "save_dir": "overfit_unet_v1_outputs",
    "seed": 42,
    "log_every": 5,
}


# =========================================================
# REPRODUCIBILITY
# =========================================================
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# =========================================================
# IMAGE HELPERS
# =========================================================
def read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_with_aspect_and_pad_rgb(img: np.ndarray, target_size: int = 1024, pad_value: int = 255) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_size, target_size, 3), pad_value, dtype=np.uint8)

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().float().cpu().permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def content_ratio(img: np.ndarray, white_threshold: int = 245) -> float:
    gray = img.mean(axis=2)
    return float((gray < white_threshold).mean())


def random_content_crop_pair(lq: np.ndarray, hq: np.ndarray, patch_size: int, min_content_ratio: float, max_attempts: int, white_threshold: int):
    h, w = hq.shape[:2]

    best = None
    best_ratio = -1.0

    for _ in range(max_attempts):
        y = random.randint(0, h - patch_size)
        x = random.randint(0, w - patch_size)

        lq_patch = lq[y:y + patch_size, x:x + patch_size]
        hq_patch = hq[y:y + patch_size, x:x + patch_size]
        ratio = content_ratio(hq_patch, white_threshold)

        if ratio > best_ratio:
            best = (lq_patch, hq_patch, ratio)
            best_ratio = ratio

        if ratio >= min_content_ratio:
            return lq_patch, hq_patch, ratio

    return best


def augment_pair(lq: np.ndarray, hq: np.ndarray):
    if random.random() < 0.5:
        lq = np.fliplr(lq).copy()
        hq = np.fliplr(hq).copy()
    if random.random() < 0.5:
        lq = np.flipud(lq).copy()
        hq = np.flipud(hq).copy()
    k = random.randint(0, 3)
    if k:
        lq = np.rot90(lq, k).copy()
        hq = np.rot90(hq, k).copy()
    return lq, hq


# =========================================================
# DATASET
# =========================================================
class TinyOverfitDataset(Dataset):
    def __init__(self, lq_dir: str, hq_dir: str, cfg: dict):
        self.cfg = cfg
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        lq_files = {Path(f).stem: f for f in os.listdir(lq_dir) if f.lower().endswith(exts)}
        hq_files = {Path(f).stem: f for f in os.listdir(hq_dir) if f.lower().endswith(exts)}
        stems = sorted(set(lq_files) & set(hq_files))[: cfg["num_images"]]

        if not stems:
            raise RuntimeError("No paired files found")

        self.pairs = []
        for stem in stems:
            lq = read_rgb(os.path.join(lq_dir, lq_files[stem]))
            hq = read_rgb(os.path.join(hq_dir, hq_files[stem]))

            lq = resize_with_aspect_and_pad_rgb(lq, cfg["image_size"], pad_value=255)
            hq = resize_with_aspect_and_pad_rgb(hq, cfg["image_size"], pad_value=255)
            self.pairs.append((lq, hq, stem))

        print(f"TinyOverfitDataset: {len(self.pairs)} images, {len(self)} patches per epoch")

    def __len__(self):
        return len(self.pairs) * self.cfg["patches_per_image"]

    def __getitem__(self, idx):
        img_idx = idx % len(self.pairs)
        lq, hq, stem = self.pairs[img_idx]

        lq_patch, hq_patch, ratio = random_content_crop_pair(
            lq,
            hq,
            patch_size=self.cfg["patch_size"],
            min_content_ratio=self.cfg["min_content_ratio"],
            max_attempts=self.cfg["max_crop_attempts"],
            white_threshold=self.cfg["white_threshold"],
        )

        lq_patch, hq_patch = augment_pair(lq_patch, hq_patch)
        return to_tensor(lq_patch), to_tensor(hq_patch), stem, torch.tensor(ratio, dtype=torch.float32)


# =========================================================
# MODEL
# =========================================================
def make_gn(ch: int):
    groups = 8 if ch >= 64 else 4
    return nn.GroupNorm(groups, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.SiLU(inplace=True),
        )
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=48, residual_scale=0.25):
        super().__init__()
        self.residual_scale = residual_scale

        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)

        self.mid = ConvBlock(base * 8, base * 8)

        self.up4 = UpBlock(base * 8, base * 8, base * 4)
        self.up3 = UpBlock(base * 4, base * 4, base * 2)
        self.up2 = UpBlock(base * 2, base * 2, base)
        self.up1 = UpBlock(base, base, base)

        self.final = nn.Sequential(
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.mid(self.pool(e4))

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        correction = self.final(d1)
        return torch.clamp(x + self.residual_scale * correction, 0.0, 1.0), correction


# =========================================================
# LOSSES AND METRICS
# =========================================================
def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def line_mask(target: torch.Tensor, white_threshold: float = 245 / 255.0) -> torch.Tensor:
    gray = target.mean(dim=1, keepdim=True)
    mask = (gray < white_threshold).float()
    # Slight dilation so near-line pixels also matter.
    mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask


def restoration_loss(pred: torch.Tensor, target: torch.Tensor):
    l1 = F.l1_loss(pred, target)

    mask = line_mask(target)
    weighted_l1 = (torch.abs(pred - target) * (1.0 + 8.0 * mask)).mean()

    edge_l1 = F.l1_loss(sobel_edges(pred), sobel_edges(target))

    loss = 0.35 * l1 + 0.45 * weighted_l1 + 0.20 * edge_l1
    return loss, {
        "l1": l1.detach(),
        "weighted_l1": weighted_l1.detach(),
        "edge_l1": edge_l1.detach(),
    }


def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target).clamp(min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def calc_line_l1(pred, target):
    mask = line_mask(target)
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)


# =========================================================
# LOGGING
# =========================================================
def log_images(logger, lq, pred, hq, correction, epoch, save_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(tensor_to_numpy(lq))
    axes[0].set_title("LQ")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_numpy(pred))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(hq))
    axes[2].set_title("HQ")
    axes[2].axis("off")

    corr_np = correction.detach().float().cpu().permute(1, 2, 0).numpy()
    corr_vis = (corr_np + 1.0) / 2.0
    axes[3].imshow(np.clip(corr_vis, 0, 1))
    axes[3].set_title("Correction, visualized")
    axes[3].axis("off")

    plt.tight_layout()
    logger.report_matplotlib_figure("Overfit diagnostics", "LQ / Pred / HQ / Correction", fig, iteration=epoch)

    path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================
# MAIN
# =========================================================
def main():
    setup_clearml_credentials()
    cfg = CONFIG
    seed_everything(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    task = Task.init(
        project_name=cfg["project_name"],
        task_name=cfg["task_name"],
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )
    cfg = task.connect(cfg)
    logger = task.get_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    logger.report_text(f"Device: {device}")

    clearml_dataset = ClearMLDataset.get(dataset_id=cfg["clearml_dataset_id"])
    local_path = clearml_dataset.get_local_copy()
    lq_dir = os.path.join(local_path, "LQ")
    hq_dir = os.path.join(local_path, "HQ")

    dataset = TinyOverfitDataset(lq_dir, hq_dir, cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    model = ResidualUNet(base=cfg["base_channels"], residual_scale=cfg["residual_scale"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_line_l1 = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_overfit_model.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_wl1 = 0.0
        total_edge = 0.0
        total_psnr = 0.0
        total_line_l1 = 0.0

        last_batch = None

        for lq, hq, _, _ in loader:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred, correction = model(lq)
                loss, parts = restoration_loss(pred, hq)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                total_loss += loss.item()
                total_l1 += parts["l1"].item()
                total_wl1 += parts["weighted_l1"].item()
                total_edge += parts["edge_l1"].item()
                total_psnr += calc_psnr(pred.float(), hq.float()).item()
                total_line_l1 += calc_line_l1(pred.float(), hq.float()).item()
                last_batch = (lq[0].detach(), pred[0].detach(), hq[0].detach(), correction[0].detach())

        n = len(loader)
        avg_loss = total_loss / n
        avg_l1 = total_l1 / n
        avg_wl1 = total_wl1 / n
        avg_edge = total_edge / n
        avg_psnr = total_psnr / n
        avg_line_l1 = total_line_l1 / n

        logger.report_scalar("Loss", "total", avg_loss, epoch)
        logger.report_scalar("Loss", "l1", avg_l1, epoch)
        logger.report_scalar("Loss", "weighted_l1", avg_wl1, epoch)
        logger.report_scalar("Loss", "edge_l1", avg_edge, epoch)
        logger.report_scalar("Metrics", "psnr_reference_only", avg_psnr, epoch)
        logger.report_scalar("Metrics", "line_l1", avg_line_l1, epoch)

        if epoch % cfg["log_every"] == 0 and last_batch is not None:
            img_path = log_images(logger, *last_batch, epoch=epoch, save_dir=cfg["save_dir"])
            task.upload_artifact(f"visual_epoch_{epoch:04d}", img_path)

        if avg_line_l1 < best_line_l1:
            best_line_l1 = avg_line_l1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_line_l1": best_line_l1,
                    "config": dict(cfg),
                },
                best_path,
            )
            task.upload_artifact("best_overfit_model", best_path)

        print(
            f"Epoch {epoch:04d}/{cfg['epochs']} | "
            f"loss={avg_loss:.6f} | l1={avg_l1:.6f} | "
            f"weighted_l1={avg_wl1:.6f} | edge={avg_edge:.6f} | "
            f"line_l1={avg_line_l1:.6f} | psnr_ref={avg_psnr:.3f}"
        )

    summary = {
        "best_line_l1": best_line_l1,
        "config": dict(cfg),
    }
    summary_path = os.path.join(cfg["save_dir"], "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    task.upload_artifact("summary", summary_path)
    task.upload_artifact("outputs_folder", cfg["save_dir"])
    task.close()


if __name__ == "__main__":
    main()
