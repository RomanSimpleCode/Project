#!/usr/bin/env python
# coding: utf-8

"""
Diagnostic overfit-test v6: Residual Attention U-Net for CAD restoration.

Why this version:
- We keep U-Net skip connections because they are important for floorplan geometry.
- We avoid the weak previous U-Net design by adding:
    1. Residual ConvBlocks
    2. Attention gates on skip connections
    3. Deep supervision residual heads at decoder levels
    4. Explicit residual supervision: correction ≈ HQ - LQ
    5. Line-weighted Charbonnier loss

Task:
    LQ -> HQ
    model predicts correction, pred = LQ + correction

Main metrics:
    line_l1
    lq_line baseline
    gain_line = lq_line - model_line

PSNR is intentionally not used as a main metric.
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
# CONFIG
# =========================================================
CONFIG = {
    "project_name": "Vosstanovlenie_tehnicheskih_sistem",
    "task_name": "diagnostic_overfit_resatt_unet_v6",
    "clearml_dataset_id": "PUT_YOUR_DATASET_ID_HERE",

    "image_size": 1024,
    "patch_size": 512,
    "num_images": 3,
    "patches_per_image": 4,
    "batch_size": 2,
    "num_workers": 2,

    # Model
    "base_channels": 64,
    "dropout": 0.0,
    "residual_scale": 1.0,

    # Training
    "epochs": 180,
    "lr": 2e-4,
    "min_lr": 1e-6,
    "weight_decay": 0.0,
    "lr_patience": 25,
    "grad_clip": 1.0,

    # Loss weights
    "w_base": 0.10,
    "w_line": 0.65,
    "w_residual": 0.45,
    "w_edge": 0.05,
    "w_deep": 0.20,
    "line_mask_weight": 12.0,

    "white_threshold": 245,
    "save_dir": "overfit_resatt_unet_v6_outputs",
    "seed": 42,
    "log_every": 10,
}


# =========================================================
# CLEARML CREDENTIALS
# =========================================================
def _safe_secret_to_str(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("value", "text", "secret", "key"):
            if k in value and isinstance(value[k], str):
                return value[k]
    return str(value)


def setup_clearml_credentials():
    api_host = str(os.environ.get("CLEARML_API_HOST", "https://api.clear.ml"))
    web_host = str(os.environ.get("CLEARML_WEB_HOST", "https://app.clear.ml"))
    files_host = str(os.environ.get("CLEARML_FILES_HOST", "https://files.clear.ml"))

    key = os.environ.get("CLEARML_API_ACCESS_KEY")
    secret = os.environ.get("CLEARML_API_SECRET_KEY")

    if not isinstance(key, str) or not key:
        key = _safe_secret_to_str(getpass("ClearML API access key: "))

    if not isinstance(secret, str) or not secret:
        secret = _safe_secret_to_str(getpass("ClearML API secret key: "))

    Task.set_credentials(
        api_host=api_host,
        web_host=web_host,
        files_host=files_host,
        key=key,
        secret=secret,
    )


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


def make_fixed_content_crops(lq: np.ndarray, hq: np.ndarray, patch_size: int, patches_per_image: int, white_threshold: int):
    h, w = hq.shape[:2]
    candidates = []

    steps_y = [0, max(0, (h - patch_size) // 2), max(0, h - patch_size)]
    steps_x = [0, max(0, (w - patch_size) // 2), max(0, w - patch_size)]

    for y in steps_y:
        for x in steps_x:
            lq_patch = lq[y:y + patch_size, x:x + patch_size]
            hq_patch = hq[y:y + patch_size, x:x + patch_size]
            ratio = content_ratio(hq_patch, white_threshold)
            candidates.append((ratio, y, x, lq_patch, hq_patch))

    rng = random.Random(12345)
    for _ in range(80):
        y = rng.randint(0, h - patch_size)
        x = rng.randint(0, w - patch_size)
        lq_patch = lq[y:y + patch_size, x:x + patch_size]
        hq_patch = hq[y:y + patch_size, x:x + patch_size]
        ratio = content_ratio(hq_patch, white_threshold)
        candidates.append((ratio, y, x, lq_patch, hq_patch))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:patches_per_image]
    return [(item[3], item[4], item[0]) for item in selected]


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

        self.samples = []
        for stem in stems:
            lq = read_rgb(os.path.join(lq_dir, lq_files[stem]))
            hq = read_rgb(os.path.join(hq_dir, hq_files[stem]))

            lq = resize_with_aspect_and_pad_rgb(lq, cfg["image_size"], pad_value=255)
            hq = resize_with_aspect_and_pad_rgb(hq, cfg["image_size"], pad_value=255)

            fixed_crops = make_fixed_content_crops(
                lq=lq,
                hq=hq,
                patch_size=cfg["patch_size"],
                patches_per_image=cfg["patches_per_image"],
                white_threshold=cfg["white_threshold"],
            )

            for crop_id, (lq_patch, hq_patch, ratio) in enumerate(fixed_crops):
                self.samples.append((lq_patch, hq_patch, f"{stem}_crop{crop_id}", ratio))

        print(f"TinyOverfitDataset: {len(stems)} images, {len(self.samples)} fixed patches total")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        lq_patch, hq_patch, stem, ratio = self.samples[idx]
        return to_tensor(lq_patch), to_tensor(hq_patch), stem, torch.tensor(ratio, dtype=torch.float32)


# =========================================================
# MODEL: RESIDUAL ATTENTION U-NET
# =========================================================
def make_gn(ch: int):
    groups = 8 if ch >= 64 else 4
    return nn.GroupNorm(groups, ch)


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = make_gn(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = make_gn(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        out = self.act(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        return self.act(out + identity)


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            make_gn(inter_ch),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            make_gn(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1),
            nn.Sigmoid(),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        att = self.psi(self.act(self.gate_proj(gate) + self.skip_proj(skip)))
        return skip * att


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.SiLU(inplace=True),
        )
        self.att = AttentionGate(out_ch, skip_ch, max(out_ch // 2, 16))
        self.conv = nn.Sequential(
            ResConvBlock(out_ch + skip_ch, out_ch, dropout=dropout),
            ResConvBlock(out_ch, out_ch, dropout=dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.att(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class ResidualAttentionUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, dropout=0.0, residual_scale=1.0):
        super().__init__()
        self.residual_scale = residual_scale

        self.enc1 = nn.Sequential(ResConvBlock(in_ch, base, dropout), ResConvBlock(base, base, dropout))
        self.enc2 = nn.Sequential(ResConvBlock(base, base * 2, dropout), ResConvBlock(base * 2, base * 2, dropout))
        self.enc3 = nn.Sequential(ResConvBlock(base * 2, base * 4, dropout), ResConvBlock(base * 4, base * 4, dropout))
        self.enc4 = nn.Sequential(ResConvBlock(base * 4, base * 8, dropout), ResConvBlock(base * 8, base * 8, dropout))

        self.pool = nn.MaxPool2d(2)

        self.mid = nn.Sequential(
            ResConvBlock(base * 8, base * 8, dropout),
            ResConvBlock(base * 8, base * 8, dropout),
        )

        self.up4 = UpBlock(base * 8, base * 8, base * 4, dropout)
        self.up3 = UpBlock(base * 4, base * 4, base * 2, dropout)
        self.up2 = UpBlock(base * 2, base * 2, base, dropout)
        self.up1 = UpBlock(base, base, base, dropout)

        self.final = nn.Conv2d(base, out_ch, 3, padding=1)

        # Deep supervision residual heads.
        self.ds2 = nn.Conv2d(base, out_ch, 3, padding=1)
        self.ds3 = nn.Conv2d(base * 2, out_ch, 3, padding=1)
        self.ds4 = nn.Conv2d(base * 4, out_ch, 3, padding=1)

        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        nn.init.zeros_(self.ds2.weight)
        nn.init.zeros_(self.ds2.bias)
        nn.init.zeros_(self.ds3.weight)
        nn.init.zeros_(self.ds3.bias)
        nn.init.zeros_(self.ds4.weight)
        nn.init.zeros_(self.ds4.bias)

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
        pred = x + self.residual_scale * correction

        ds = []
        for feat, head in [(d2, self.ds2), (d3, self.ds3), (d4, self.ds4)]:
            corr = head(feat)
            corr = F.interpolate(corr, size=x.shape[-2:], mode="bilinear", align_corners=False)
            ds.append(corr)

        return pred, correction, ds


# =========================================================
# LOSSES AND METRICS
# =========================================================
def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt(x * x + eps * eps)


def line_mask(target: torch.Tensor, white_threshold: float = 245 / 255.0) -> torch.Tensor:
    gray = target.mean(dim=1, keepdim=True)
    mask = (gray < white_threshold).float()
    mask = F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)
    return mask


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    gray = x.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def calc_line_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = line_mask(target)
    return (torch.abs(pred - target) * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)


def weighted_l1_metric(pred: torch.Tensor, target: torch.Tensor, mask_weight: float = 12.0) -> torch.Tensor:
    mask = line_mask(target)
    return (torch.abs(pred - target) * (1.0 + mask_weight * mask)).mean()


def residual_charbonnier_loss(correction, true_residual, mask, mask_weight):
    return (charbonnier(correction - true_residual) * (1.0 + mask_weight * mask)).mean()


def restoration_loss(pred, hq, lq, correction, ds_corrections, cfg):
    mask = line_mask(hq)
    diff = pred - hq
    true_residual = hq - lq

    base = charbonnier(diff).mean()
    line = (charbonnier(diff) * (1.0 + cfg["line_mask_weight"] * mask)).mean()
    residual = residual_charbonnier_loss(correction, true_residual, mask, cfg["line_mask_weight"])
    edge = F.l1_loss(sobel_edges(pred), sobel_edges(hq))

    deep = torch.zeros_like(base)
    for ds_corr in ds_corrections:
        deep = deep + residual_charbonnier_loss(ds_corr, true_residual, mask, cfg["line_mask_weight"])
    deep = deep / max(1, len(ds_corrections))

    loss = (
        cfg["w_base"] * base
        + cfg["w_line"] * line
        + cfg["w_residual"] * residual
        + cfg["w_edge"] * edge
        + cfg["w_deep"] * deep
    )

    return loss, {
        "l1": F.l1_loss(pred, hq).detach(),
        "weighted_l1": weighted_l1_metric(pred, hq, cfg["line_mask_weight"]).detach(),
        "residual": residual.detach(),
        "edge_ref": edge.detach(),
        "deep": deep.detach(),
    }


# =========================================================
# LOGGING
# =========================================================
def log_images(logger, lq, pred, hq, correction, epoch, save_dir):
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))

    axes[0].imshow(tensor_to_numpy(lq))
    axes[0].set_title("LQ")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_numpy(pred))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(hq))
    axes[2].set_title("HQ")
    axes[2].axis("off")

    diff = torch.abs(pred - hq).mean(dim=0, keepdim=False)
    axes[3].imshow(diff.detach().float().cpu().numpy(), cmap="gray")
    axes[3].set_title("Abs diff")
    axes[3].axis("off")

    corr_np = correction.detach().float().cpu().permute(1, 2, 0).numpy()
    corr_vis = np.clip((corr_np * 6.0) + 0.5, 0, 1)
    axes[4].imshow(corr_vis)
    axes[4].set_title("Correction x6")
    axes[4].axis("off")

    plt.tight_layout()
    logger.report_matplotlib_figure("Overfit diagnostics", "LQ / Pred / HQ / Diff / Correction", fig, iteration=epoch)

    path = os.path.join(save_dir, f"epoch_{epoch:04d}.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# =========================================================
# MAIN
# =========================================================
def main():
    setup_clearml_credentials()
    cfg = dict(CONFIG)
    seed_everything(cfg["seed"])
    os.makedirs(cfg["save_dir"], exist_ok=True)

    task = Task.init(
        project_name=cfg["project_name"],
        task_name=cfg["task_name"],
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )
    task.connect(cfg)
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

    model = ResidualAttentionUNet(
        base=cfg["base_channels"],
        dropout=cfg["dropout"],
        residual_scale=cfg["residual_scale"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    logger.report_text(f"Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=cfg["lr_patience"],
        min_lr=cfg["min_lr"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_line_l1 = float("inf")
    best_path = os.path.join(cfg["save_dir"], "best_overfit_resatt_unet.pth")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        sums = {
            "loss": 0.0,
            "l1": 0.0,
            "weighted_l1": 0.0,
            "residual": 0.0,
            "edge_ref": 0.0,
            "deep": 0.0,
            "line_l1": 0.0,
            "lq_line": 0.0,
            "gain_line": 0.0,
            "gain_weighted": 0.0,
        }

        last_batch = None

        for lq, hq, _, _ in loader:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pred, correction, ds = model(lq)
                loss, parts = restoration_loss(pred, hq, lq, correction, ds, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                model_line = calc_line_l1(pred.float(), hq.float()).item()
                lq_line = calc_line_l1(lq.float(), hq.float()).item()
                model_weighted = weighted_l1_metric(pred.float(), hq.float(), cfg["line_mask_weight"]).item()
                lq_weighted = weighted_l1_metric(lq.float(), hq.float(), cfg["line_mask_weight"]).item()

                sums["loss"] += loss.item()
                sums["l1"] += parts["l1"].item()
                sums["weighted_l1"] += parts["weighted_l1"].item()
                sums["residual"] += parts["residual"].item()
                sums["edge_ref"] += parts["edge_ref"].item()
                sums["deep"] += parts["deep"].item()
                sums["line_l1"] += model_line
                sums["lq_line"] += lq_line
                sums["gain_line"] += (lq_line - model_line)
                sums["gain_weighted"] += (lq_weighted - model_weighted)

                last_batch = (lq[0].detach(), pred[0].detach(), hq[0].detach(), correction[0].detach())

        n = len(loader)
        avg = {k: v / n for k, v in sums.items()}
        scheduler.step(avg["line_l1"])

        logger.report_scalar("Loss", "total", avg["loss"], epoch)
        logger.report_scalar("Loss", "l1", avg["l1"], epoch)
        logger.report_scalar("Loss", "weighted_l1", avg["weighted_l1"], epoch)
        logger.report_scalar("Loss", "residual", avg["residual"], epoch)
        logger.report_scalar("Loss", "edge_ref", avg["edge_ref"], epoch)
        logger.report_scalar("Loss", "deep", avg["deep"], epoch)
        logger.report_scalar("Metrics", "model_line_l1", avg["line_l1"], epoch)
        logger.report_scalar("Metrics", "lq_line_l1_baseline", avg["lq_line"], epoch)
        logger.report_scalar("Metrics", "gain_line_l1_positive_is_good", avg["gain_line"], epoch)
        logger.report_scalar("Metrics", "gain_weighted_l1_positive_is_good", avg["gain_weighted"], epoch)
        logger.report_scalar("LR", "current", optimizer.param_groups[0]["lr"], epoch)

        if epoch % cfg["log_every"] == 0 and last_batch is not None:
            img_path = log_images(logger, *last_batch, epoch=epoch, save_dir=cfg["save_dir"])
            task.upload_artifact(f"visual_epoch_{epoch:04d}", img_path)

        if avg["line_l1"] < best_line_l1:
            best_line_l1 = avg["line_l1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "best_line_l1": best_line_l1,
                    "config": dict(cfg),
                    "architecture": "ResidualAttentionUNet_deep_supervision",
                },
                best_path,
            )
            task.upload_artifact("best_overfit_resatt_unet", best_path)

        print(
            f"Epoch {epoch:04d}/{cfg['epochs']} | "
            f"loss={avg['loss']:.6f} | "
            f"l1={avg['l1']:.6f} | "
            f"weighted_l1={avg['weighted_l1']:.6f} | "
            f"residual={avg['residual']:.6f} | "
            f"deep={avg['deep']:.6f} | "
            f"edge_ref={avg['edge_ref']:.6f} | "
            f"line_l1={avg['line_l1']:.6f} | "
            f"lq_line={avg['lq_line']:.6f} | "
            f"gain_line={avg['gain_line']:+.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
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
