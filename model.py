#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install clearml pytorch-msssim')


# In[ ]:


import os
import random
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from pytorch_msssim import ssim
from clearml import Task, Dataset as ClearMLDataset


# =========================================================
# CLEARML CREDENTIALS
# =========================================================
Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key="ZCGVG8PSQJOOZNASXTHNO3MLHWXK6G",
    secret="KQ46iRWMS_IRDUuB8BggNUZwj-3e_0CvEWmQXJU3pIWfanybc6smB4tfxxGKbQ1qSKI",
)


# =========================================================
# CLEARML TASK
# =========================================================
old_task = Task.current_task()
if old_task is not None:
    old_task.close()

task = Task.init(
    project_name="Vosstanovlenie_tehnicheskih_sistem",
    task_name="V24_Residual_UNet_FullDataset_MSE",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)

logger = task.get_logger()


# =========================================================
# CONFIG
# =========================================================
config = {
    # Полный исходный датасет — лучший результат был именно на нём
    "clearml_dataset_id": "3ea1e9f808034406bdf383ff1bbb32f4",

    # Data
    "image_size": 1000,
    "patch_size": 512,
    "test_split": 0.20,
    "subset_size": -1,

    "patches_per_image": 4,
    "val_patches_per_image": 2,

    "min_patch_content_ratio": 0.01,
    "white_threshold": 245,
    "max_crop_attempts": 80,
    "pad_value": 255,

    # Dataloader
    "batch_size": 4,
    "num_workers": 2,

    # Model
    "base_channels": 32,
    "dropout_rate": 0.0,

    # Обучаем мягкую поправку, потом подбираем scale
    "train_residual_scale": 0.02,

    # Training
    "epochs": 80,
    "lr": 1e-4,
    "weight_decay": 1e-6,
    "early_stopping_patience": 12,

    # Scale sweep after training
    "sweep_scales": [0.02, 0.05, 0.07, 0.10, 0.12],

    # Logging / saving
    "seed": 42,
    "save_dir": "v24_residual_unet_full_dataset",
    "log_images_every": 1,
    "save_epoch_every": 5,
}

config = task.connect(config)

os.makedirs(config["save_dir"], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.report_text(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# SEED
# =========================================================
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = True


# =========================================================
# DATASET DOWNLOAD
# =========================================================
dataset_artifact = ClearMLDataset.get(dataset_id=config["clearml_dataset_id"])
local_path = dataset_artifact.get_local_copy()

print(f"Dataset: {local_path}")
logger.report_text(f"Dataset path: {local_path}")

lq_dir = os.path.join(local_path, "LQ")
hq_dir = os.path.join(local_path, "HQ")

assert os.path.isdir(lq_dir), f"LQ folder not found: {lq_dir}"
assert os.path.isdir(hq_dir), f"HQ folder not found: {hq_dir}"


# =========================================================
# HELPERS
# =========================================================
def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Failed to read image: {path}")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize_with_aspect_and_pad_rgb(img, target_size=1000, pad_value=255):
    h, w = img.shape[:2]

    scale = min(target_size / w, target_size / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.full(
        (target_size, target_size, 3),
        pad_value,
        dtype=np.uint8
    )

    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2

    canvas[top:top + new_h, left:left + new_w] = resized

    return canvas


def calc_content_ratio_rgb(img, white_threshold=245):
    gray = img.mean(axis=2)
    return float((gray < white_threshold).mean())


def random_crop_pair(lq, hq, patch_size):
    h, w = hq.shape[:2]

    if h < patch_size or w < patch_size:
        lq = cv2.resize(lq, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        hq = cv2.resize(hq, (patch_size, patch_size), interpolation=cv2.INTER_AREA)
        return lq, hq

    y = random.randint(0, h - patch_size)
    x = random.randint(0, w - patch_size)

    return (
        lq[y:y + patch_size, x:x + patch_size],
        hq[y:y + patch_size, x:x + patch_size],
    )


def fixed_crop_pair(lq, hq, patch_size, crop_id):
    h, w = hq.shape[:2]

    positions = [
        (0, 0),
        (0, w - patch_size),
        (h - patch_size, 0),
        (h - patch_size, w - patch_size),
        ((h - patch_size) // 2, (w - patch_size) // 2),
        (0, (w - patch_size) // 2),
        ((h - patch_size) // 2, 0),
        ((h - patch_size) // 2, w - patch_size),
        (h - patch_size, (w - patch_size) // 2),
    ]

    y, x = positions[crop_id % len(positions)]

    y = max(0, min(y, h - patch_size))
    x = max(0, min(x, w - patch_size))

    return (
        lq[y:y + patch_size, x:x + patch_size],
        hq[y:y + patch_size, x:x + patch_size],
    )


def augment_pair(lq, hq):
    if random.random() < 0.5:
        lq = np.fliplr(lq).copy()
        hq = np.fliplr(hq).copy()

    if random.random() < 0.5:
        lq = np.flipud(lq).copy()
        hq = np.flipud(hq).copy()

    k = random.randint(0, 3)

    if k > 0:
        lq = np.rot90(lq, k).copy()
        hq = np.rot90(hq, k).copy()

    return lq, hq


def to_tensor(img):
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def tensor_to_numpy(t):
    arr = (
        t.detach()
        .float()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def sharpen_tensor(lq):
    outs = []

    for i in range(lq.shape[0]):
        img_np = tensor_to_numpy(lq[i])
        blur = cv2.GaussianBlur(img_np, (0, 0), sigmaX=1.0)
        sharp = cv2.addWeighted(img_np, 1.5, blur, -0.5, 0)
        sharp = np.clip(sharp, 0.0, 1.0).astype(np.float32)
        sharp_t = torch.from_numpy(sharp).permute(2, 0, 1)
        outs.append(sharp_t)

    return torch.stack(outs, dim=0).to(lq.device)


# =========================================================
# DATASET
# =========================================================
class ContentPatchDataset(Dataset):
    def __init__(
        self,
        lq_dir,
        hq_dir,
        indices,
        image_size=1000,
        patch_size=512,
        patches_per_image=4,
        augment=True,
        train=True,
        min_content_ratio=0.01,
        white_threshold=245,
        max_crop_attempts=80,
        pad_value=255,
    ):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.indices = list(indices)

        self.image_size = image_size
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.augment = augment
        self.train = train
        self.min_content_ratio = min_content_ratio
        self.white_threshold = white_threshold
        self.max_crop_attempts = max_crop_attempts
        self.pad_value = pad_value

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

        lq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(lq_dir)
            if f.lower().endswith(exts)
        }

        hq_files = {
            os.path.splitext(f)[0]: f
            for f in os.listdir(hq_dir)
            if f.lower().endswith(exts)
        }

        common_stems = sorted(set(lq_files.keys()) & set(hq_files.keys()))
        self.files = [(lq_files[s], hq_files[s]) for s in common_stems]

        if len(self.files) == 0:
            raise RuntimeError("No paired LQ/HQ images found")

        print(
            f"ContentPatchDataset | images={len(self.indices)} | "
            f"patches_per_image={self.patches_per_image} | "
            f"train={self.train} | augment={self.augment}"
        )

    def __len__(self):
        return len(self.indices) * self.patches_per_image

    def load_pair_by_real_index(self, real_idx):
        lq_fname, hq_fname = self.files[real_idx]

        lq = read_rgb(os.path.join(self.lq_dir, lq_fname))
        hq = read_rgb(os.path.join(self.hq_dir, hq_fname))

        lq = resize_with_aspect_and_pad_rgb(
            lq,
            target_size=self.image_size,
            pad_value=self.pad_value
        )

        hq = resize_with_aspect_and_pad_rgb(
            hq,
            target_size=self.image_size,
            pad_value=self.pad_value
        )

        return lq, hq, lq_fname

    def get_content_patch_train(self, lq, hq):
        best_lq_patch = None
        best_hq_patch = None
        best_ratio = -1.0

        for _ in range(self.max_crop_attempts):
            lq_patch, hq_patch = random_crop_pair(lq, hq, self.patch_size)

            ratio = calc_content_ratio_rgb(
                hq_patch,
                white_threshold=self.white_threshold
            )

            if ratio > best_ratio:
                best_ratio = ratio
                best_lq_patch = lq_patch
                best_hq_patch = hq_patch

            if ratio >= self.min_content_ratio:
                return lq_patch, hq_patch, ratio

        return best_lq_patch, best_hq_patch, best_ratio

    def get_content_patch_val(self, lq, hq, crop_id):
        best_lq_patch = None
        best_hq_patch = None
        best_ratio = -1.0

        for offset in range(9):
            lq_patch, hq_patch = fixed_crop_pair(
                lq,
                hq,
                self.patch_size,
                crop_id + offset
            )

            ratio = calc_content_ratio_rgb(
                hq_patch,
                white_threshold=self.white_threshold
            )

            if ratio > best_ratio:
                best_ratio = ratio
                best_lq_patch = lq_patch
                best_hq_patch = hq_patch

        if best_ratio >= self.min_content_ratio:
            return best_lq_patch, best_hq_patch, best_ratio

        for _ in range(20):
            lq_patch, hq_patch = random_crop_pair(lq, hq, self.patch_size)

            ratio = calc_content_ratio_rgb(
                hq_patch,
                white_threshold=self.white_threshold
            )

            if ratio > best_ratio:
                best_ratio = ratio
                best_lq_patch = lq_patch
                best_hq_patch = hq_patch

        return best_lq_patch, best_hq_patch, best_ratio

    def __getitem__(self, idx):
        image_pos = idx // self.patches_per_image
        crop_id = idx % self.patches_per_image

        real_idx = self.indices[image_pos]
        lq, hq, fname = self.load_pair_by_real_index(real_idx)

        if self.train:
            lq_patch, hq_patch, content_ratio = self.get_content_patch_train(lq, hq)

            if self.augment:
                lq_patch, hq_patch = augment_pair(lq_patch, hq_patch)

        else:
            lq_patch, hq_patch, content_ratio = self.get_content_patch_val(
                lq,
                hq,
                crop_id
            )

        return (
            to_tensor(lq_patch),
            to_tensor(hq_patch),
            fname,
            torch.tensor(content_ratio, dtype=torch.float32)
        )


# =========================================================
# RESIDUAL U-NET MODEL
# =========================================================
def make_gn(ch):
    if ch >= 128:
        return nn.GroupNorm(8, ch)
    return nn.GroupNorm(4, ch)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            make_gn(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResidualUNet(nn.Module):
    """
    Residual U-Net:
    correction = UNet(input)
    output = input + residual_scale * correction
    """

    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        base=32,
        dropout_rate=0.0,
        residual_scale=0.02,
    ):
        super().__init__()

        self.residual_scale = residual_scale

        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConvBlock(base * 8, base * 8),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        # Decoder with skip connections
        self.up4 = UpBlock(base * 8, base * 8, base * 4)
        self.up3 = UpBlock(base * 4, base * 4, base * 2)
        self.up2 = UpBlock(base * 2, base * 2, base)
        self.up1 = UpBlock(base, base, base)

        self.final = nn.Sequential(
            nn.Conv2d(base, out_ch, 3, padding=1),
            nn.Tanh(),
        )

        self.init_final_zero()

    def init_final_zero(self):
        final_conv = self.final[0]
        nn.init.zeros_(final_conv.weight)

        if final_conv.bias is not None:
            nn.init.zeros_(final_conv.bias)

    def forward(self, x):
        original_size = x.shape[-2:]

        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        b = self.bottleneck(p4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        if d1.shape[-2:] != original_size:
            d1 = F.interpolate(
                d1,
                size=original_size,
                mode="bilinear",
                align_corners=False
            )

        correction = self.final(d1)

        out = x + self.residual_scale * correction
        out = torch.clamp(out, 0.0, 1.0)

        return out


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight,
            a=0.1,
            mode="fan_out",
            nonlinearity="leaky_relu"
        )

        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.GroupNorm):
        if m.weight is not None:
            nn.init.ones_(m.weight)

        if m.bias is not None:
            nn.init.zeros_(m.bias)


# =========================================================
# DATA SPLIT
# =========================================================
exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

lq_files = {
    os.path.splitext(f)[0]: f
    for f in os.listdir(lq_dir)
    if f.lower().endswith(exts)
}

hq_files = {
    os.path.splitext(f)[0]: f
    for f in os.listdir(hq_dir)
    if f.lower().endswith(exts)
}

common_stems = sorted(set(lq_files.keys()) & set(hq_files.keys()))
dataset_len = len(common_stems)

if dataset_len == 0:
    raise RuntimeError("No paired images found")

print(f"Total paired images: {dataset_len}")

if config["subset_size"] is None or config["subset_size"] <= 0 or config["subset_size"] > dataset_len:
    selected_indices = list(range(dataset_len))
    print(f"Using full dataset: {dataset_len} images")
else:
    selected_indices = random.sample(range(dataset_len), config["subset_size"])
    print(f"Using subset: {len(selected_indices)} / {dataset_len}")

total = len(selected_indices)
test_size = max(1, int(total * config["test_split"]))
train_size = total - test_size

split_gen = torch.Generator().manual_seed(config["seed"])

train_selected, test_selected = torch.utils.data.random_split(
    selected_indices,
    [train_size, test_size],
    generator=split_gen
)

train_indices = list(train_selected)
test_indices = list(test_selected)

train_dataset = ContentPatchDataset(
    lq_dir=lq_dir,
    hq_dir=hq_dir,
    indices=train_indices,
    image_size=config["image_size"],
    patch_size=config["patch_size"],
    patches_per_image=config["patches_per_image"],
    augment=True,
    train=True,
    min_content_ratio=config["min_patch_content_ratio"],
    white_threshold=config["white_threshold"],
    max_crop_attempts=config["max_crop_attempts"],
    pad_value=config["pad_value"],
)

test_dataset = ContentPatchDataset(
    lq_dir=lq_dir,
    hq_dir=hq_dir,
    indices=test_indices,
    image_size=config["image_size"],
    patch_size=config["patch_size"],
    patches_per_image=config["val_patches_per_image"],
    augment=False,
    train=False,
    min_content_ratio=config["min_patch_content_ratio"],
    white_threshold=config["white_threshold"],
    max_crop_attempts=config["max_crop_attempts"],
    pad_value=config["pad_value"],
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True,
    drop_last=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
    drop_last=False,
)

print(f"Train images: {train_size} | Test images: {test_size}")
print(f"Train patches: {len(train_dataset)} | Test patches: {len(test_dataset)}")

logger.report_text(f"Train images: {train_size} | Test images: {test_size}")
logger.report_text(f"Train patches: {len(train_dataset)} | Test patches: {len(test_dataset)}")


# =========================================================
# MODEL / OPTIMIZER
# =========================================================
model = ResidualUNet(
    in_ch=3,
    out_ch=3,
    base=config["base_channels"],
    dropout_rate=config["dropout_rate"],
    residual_scale=config["train_residual_scale"],
)

model.apply(init_weights)
model.init_final_zero()
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {total_params:,}")
logger.report_text(f"Parameters: {total_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["lr"],
    weight_decay=config["weight_decay"],
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4,
)

scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
criterion = nn.MSELoss()


# =========================================================
# PATHS
# =========================================================
epochs_ckpt_dir = os.path.join(config["save_dir"], "epoch_checkpoints")
os.makedirs(epochs_ckpt_dir, exist_ok=True)

best_model_path = os.path.join(config["save_dir"], "best_model.pth")
last_model_path = os.path.join(config["save_dir"], "last_model.pth")


def save_checkpoint(path, epoch, best_psnr):
    checkpoint = {
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_psnr": best_psnr,
        "config": dict(config),
        "architecture": "ResidualUNet",
    }

    torch.save(checkpoint, path)


def upload_artifact_safe(name, path):
    task.upload_artifact(name, path)
    print(f"Uploaded artifact: {name}")


def log_comparison(lq, pred, hq, iteration, series, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(tensor_to_numpy(lq))
    axes[0].set_title("LQ Input")
    axes[0].axis("off")

    axes[1].imshow(tensor_to_numpy(pred))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(tensor_to_numpy(hq))
    axes[2].set_title("HQ Target")
    axes[2].axis("off")

    plt.tight_layout()

    logger.report_matplotlib_figure(
        title=title,
        series=series,
        figure=fig,
        iteration=iteration
    )

    vis_dir = os.path.join(config["save_dir"], "visual_examples")
    os.makedirs(vis_dir, exist_ok=True)

    save_path = os.path.join(vis_dir, f"epoch_{iteration + 1:03d}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

    task.upload_artifact(
        name=f"visual_epoch_{iteration + 1:03d}",
        artifact_object=save_path
    )

    plt.close(fig)


# =========================================================
# TRAINING LOOP
# =========================================================
best_psnr = 0.0
epochs_no_improve = 0
last_completed_epoch = 0

for epoch in range(config["epochs"]):
    model.train()

    train_mse_sum = 0.0
    train_content_sum = 0.0

    for lq, hq, _, content_ratio in train_loader:
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(lq)
            loss = criterion(pred, hq)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        train_mse_sum += loss.item()
        train_content_sum += content_ratio.float().mean().item()

    n_train = len(train_loader)

    train_mse = train_mse_sum / n_train
    train_content = train_content_sum / n_train

    logger.report_scalar("Loss", "train_MSE", train_mse, epoch)
    logger.report_scalar("Content", "train_content_ratio", train_content, epoch)

    # =====================================================
    # VALIDATION
    # =====================================================
    model.eval()

    test_psnr_sum = 0.0
    test_lq_psnr_sum = 0.0
    test_sharp_psnr_sum = 0.0

    test_mse_sum = 0.0
    test_lq_mse_sum = 0.0
    test_sharp_mse_sum = 0.0

    test_ssim_sum = 0.0
    test_lq_ssim_sum = 0.0
    test_sharp_ssim_sum = 0.0

    test_sample = None

    with torch.no_grad():
        for i, (lq, hq, _, _) in enumerate(test_loader):
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                pred = model(lq)

            pred_f = pred.float()
            hq_f = hq.float()
            lq_f = lq.float()
            sharp_f = sharpen_tensor(lq_f)

            test_mse_sum += F.mse_loss(pred_f, hq_f).item()
            test_lq_mse_sum += F.mse_loss(lq_f, hq_f).item()
            test_sharp_mse_sum += F.mse_loss(sharp_f, hq_f).item()

            test_psnr_sum += calc_psnr(pred_f, hq_f).item()
            test_lq_psnr_sum += calc_psnr(lq_f, hq_f).item()
            test_sharp_psnr_sum += calc_psnr(sharp_f, hq_f).item()

            test_ssim_sum += ssim(pred_f, hq_f, data_range=1.0, size_average=True).item()
            test_lq_ssim_sum += ssim(lq_f, hq_f, data_range=1.0, size_average=True).item()
            test_sharp_ssim_sum += ssim(sharp_f, hq_f, data_range=1.0, size_average=True).item()

            if i == 0:
                test_sample = (
                    lq[0].detach().float().cpu(),
                    pred[0].detach().float().cpu(),
                    hq[0].detach().float().cpu(),
                )

    n_test = len(test_loader)

    test_psnr = test_psnr_sum / n_test
    test_lq_psnr = test_lq_psnr_sum / n_test
    test_sharp_psnr = test_sharp_psnr_sum / n_test

    test_mse = test_mse_sum / n_test
    test_lq_mse = test_lq_mse_sum / n_test
    test_sharp_mse = test_sharp_mse_sum / n_test

    test_ssim = test_ssim_sum / n_test
    test_lq_ssim = test_lq_ssim_sum / n_test
    test_sharp_ssim = test_sharp_ssim_sum / n_test

    gain_lq = test_psnr - test_lq_psnr
    gain_sharp = test_psnr - test_sharp_psnr

    scheduler.step(test_psnr)
    current_lr = optimizer.param_groups[0]["lr"]

    logger.report_scalar("PSNR", "model", test_psnr, epoch)
    logger.report_scalar("PSNR", "LQ_baseline", test_lq_psnr, epoch)
    logger.report_scalar("PSNR", "sharpen_baseline", test_sharp_psnr, epoch)
    logger.report_scalar("PSNR", "gain_over_LQ", gain_lq, epoch)
    logger.report_scalar("PSNR", "gain_over_sharpen", gain_sharp, epoch)

    logger.report_scalar("MSE", "model", test_mse, epoch)
    logger.report_scalar("MSE", "LQ_baseline", test_lq_mse, epoch)
    logger.report_scalar("MSE", "sharpen_baseline", test_sharp_mse, epoch)

    logger.report_scalar("SSIM", "model", test_ssim, epoch)
    logger.report_scalar("SSIM", "LQ_baseline", test_lq_ssim, epoch)
    logger.report_scalar("SSIM", "sharpen_baseline", test_sharp_ssim, epoch)
    logger.report_scalar("LR", "learning_rate", current_lr, epoch)

    if test_sample is not None and ((epoch + 1) % config["log_images_every"] == 0):
        log_comparison(
            *test_sample,
            iteration=epoch,
            series="residual_unet_test_examples",
            title=(
                f"Epoch {epoch + 1} | "
                f"Model {test_psnr:.3f} | "
                f"LQ {test_lq_psnr:.3f} | "
                f"Sharp {test_sharp_psnr:.3f} | "
                f"Gain {gain_lq:+.3f}"
            )
        )

    print(
        f"Epoch {epoch + 1:03d}/{config['epochs']} | "
        f"LR: {current_lr:.2e} | "
        f"Train MSE: {train_mse:.7f} | "
        f"Test MSE: {test_mse:.7f} | "
        f"LQ PSNR: {test_lq_psnr:.3f} | "
        f"Sharp PSNR: {test_sharp_psnr:.3f} | "
        f"Model PSNR: {test_psnr:.3f} | "
        f"Gain LQ: {gain_lq:+.4f} | "
        f"Gain Sharp: {gain_sharp:+.4f} | "
        f"SSIM: {test_ssim:.4f}"
    )

    last_completed_epoch = epoch + 1

    save_checkpoint(last_model_path, epoch, best_psnr)

    if (epoch + 1) % config["save_epoch_every"] == 0:
        upload_artifact_safe("last_model", last_model_path)

        epoch_ckpt_path = os.path.join(
            epochs_ckpt_dir,
            f"epoch_{epoch + 1:03d}.pth"
        )

        save_checkpoint(epoch_ckpt_path, epoch, best_psnr)
        upload_artifact_safe(f"epoch_{epoch + 1:03d}", epoch_ckpt_path)

    if test_psnr > best_psnr:
        best_psnr = test_psnr
        epochs_no_improve = 0

        save_checkpoint(best_model_path, epoch, best_psnr)
        upload_artifact_safe("best_model", best_model_path)

        print(f"  >> New best PSNR: {best_psnr:.3f}")

    else:
        epochs_no_improve += 1

        print(
            f"  No improvement: {epochs_no_improve}/"
            f"{config['early_stopping_patience']}"
        )

        if epochs_no_improve >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break


# =========================================================
# FINAL SCALE SWEEP
# =========================================================
print("=" * 80)
print("FINAL SCALE SWEEP")
print("=" * 80)

checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()

sweep_results = {}

for scale in config["sweep_scales"]:
    model.residual_scale = float(scale)

    psnr_sum = 0.0
    lq_psnr_sum = 0.0
    sharp_psnr_sum = 0.0

    ssim_sum = 0.0
    lq_ssim_sum = 0.0
    sharp_ssim_sum = 0.0

    with torch.no_grad():
        for lq, hq, _, _ in test_loader:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            pred = model(lq).float()

            hq_f = hq.float()
            lq_f = lq.float()
            sharp_f = sharpen_tensor(lq_f)

            psnr_sum += calc_psnr(pred, hq_f).item()
            lq_psnr_sum += calc_psnr(lq_f, hq_f).item()
            sharp_psnr_sum += calc_psnr(sharp_f, hq_f).item()

            ssim_sum += ssim(pred, hq_f, data_range=1.0, size_average=True).item()
            lq_ssim_sum += ssim(lq_f, hq_f, data_range=1.0, size_average=True).item()
            sharp_ssim_sum += ssim(sharp_f, hq_f, data_range=1.0, size_average=True).item()

    n = len(test_loader)

    result = {
        "scale": float(scale),
        "psnr": psnr_sum / n,
        "lq_psnr": lq_psnr_sum / n,
        "sharp_psnr": sharp_psnr_sum / n,
        "ssim": ssim_sum / n,
        "lq_ssim": lq_ssim_sum / n,
        "sharp_ssim": sharp_ssim_sum / n,
    }

    result["gain_lq"] = result["psnr"] - result["lq_psnr"]
    result["gain_sharp"] = result["psnr"] - result["sharp_psnr"]

    sweep_results[str(scale)] = result

    logger.report_scalar("FinalScaleSweep/PSNR", f"scale_{scale}", result["psnr"], 0)
    logger.report_scalar("FinalScaleSweep/Gain_LQ", f"scale_{scale}", result["gain_lq"], 0)
    logger.report_scalar("FinalScaleSweep/Gain_Sharp", f"scale_{scale}", result["gain_sharp"], 0)
    logger.report_scalar("FinalScaleSweep/SSIM", f"scale_{scale}", result["ssim"], 0)

    print(
        f"scale={scale:.3f} | "
        f"PSNR={result['psnr']:.4f} | "
        f"Gain LQ={result['gain_lq']:+.4f} | "
        f"Gain Sharp={result['gain_sharp']:+.4f} | "
        f"SSIM={result['ssim']:.4f}"
    )


best_sweep = max(sweep_results.values(), key=lambda x: x["psnr"])

summary = {
    "architecture": "Residual U-Net",
    "dataset_id": config["clearml_dataset_id"],
    "best_train_psnr_at_scale_0.02": best_psnr,
    "last_epoch_completed": last_completed_epoch,
    "sweep_results": sweep_results,
    "best_sweep": best_sweep,
    "config": dict(config),
}

summary_path = os.path.join(config["save_dir"], "summary.json")

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

task.upload_artifact("summary", summary_path)
task.upload_artifact("outputs_folder", config["save_dir"])

print()
print("=" * 80)
print("DONE")
print(f"Best training PSNR at scale 0.02: {best_psnr:.4f}")
print(
    f"Best sweep: scale={best_sweep['scale']} | "
    f"PSNR={best_sweep['psnr']:.4f} | "
    f"Gain LQ={best_sweep['gain_lq']:+.4f} | "
    f"Gain Sharp={best_sweep['gain_sharp']:+.4f} | "
    f"SSIM={best_sweep['ssim']:.4f}"
)
print("=" * 80)

task.close()

