#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_msssim import ssim

from clearml import Task, Dataset as ClearMLDataset


# =========================================================
# CLEARML
# =========================================================
Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key="ZCGVG8PSQJOOZNASXTHNO3MLHWXK6G",
    secret="KQ46iRWMS_IRDUuB8BggNUZwj-3e_0CvEWmQXJU3pIWfanybc6smB4tfxxGKbQ1qSKI",
)

task = Task.init(
    project_name="Vosstanovlenie_tehnicheskih_sistem",
    task_name="Please_work",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)
logger = task.get_logger()


# =========================================================
# CONFIG
# =========================================================
config = {
    "clearml_dataset_id": "3ea1e9f808034406bdf383ff1bbb32f4",

    # Data
    "image_size": 1000,
    "test_split": 0.20,
    "batch_size": 4,
    "num_workers": 2,
    "batch_size": 8,  # патчи 256x256 кушают меньше памяти

    # Model
    "dropout_rate": 0.3,

    # Training
    "epochs": 5,               # для теста, потом увеличишь
    "lr": 5e-5,
    "seed": 42,
    "save_dir": "outputs",
    "early_stopping_patience": 5,
}

config = task.connect(config)
os.makedirs(config["save_dir"], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.report_text(f"GPU: {torch.cuda.get_device_name(0)}")


# =========================================================
# DOWNLOAD DATASET FROM CLEARML
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
# DATASET
# =========================================================
class FloorPlanDataset(Dataset):
    """Патчи 256x256, гарантированно содержащие линии/структуру."""

    def __init__(self, lq_dir, hq_dir, patch_size=256, min_line_pct=0.02, augment=True):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.patch_size = patch_size
        self.min_line_pct = min_line_pct
        self.augment = augment

        exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        lq_files = {os.path.splitext(f)[0]: f for f in os.listdir(lq_dir) if f.lower().endswith(exts)}
        hq_files = {os.path.splitext(f)[0]: f for f in os.listdir(hq_dir) if f.lower().endswith(exts)}
        common_stems = sorted(set(lq_files.keys()) & set(hq_files.keys()))
        self.pairs = [(lq_files[s], hq_files[s]) for s in common_stems]

        if len(self.pairs) == 0:
            raise RuntimeError("No paired LQ/HQ images found")
        print(f"Paired images: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs) * 20  # 20 патчей с каждого изображения

    def __getitem__(self, idx):
        pair_idx = idx % len(self.pairs)
        lq_fname, hq_fname = self.pairs[pair_idx]

        # Загрузка
        lq = cv2.imread(os.path.join(self.lq_dir, lq_fname), cv2.IMREAD_COLOR)
        hq = cv2.imread(os.path.join(self.hq_dir, hq_fname), cv2.IMREAD_COLOR)
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        hq = cv2.cvtColor(hq, cv2.COLOR_BGR2RGB)

        # Ресайз если не 1000
        if lq.shape[0] != 1000 or lq.shape[1] != 1000:
            lq = cv2.resize(lq, (1000, 1000))
            hq = cv2.resize(hq, (1000, 1000))

        h, w, _ = lq.shape
        ps = self.patch_size

        # Ищем патч с линиями (до 50 попыток)
        for _ in range(50):
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)

            hq_patch = hq[y:y+ps, x:x+ps]
            gray = cv2.cvtColor(hq_patch, cv2.COLOR_RGB2GRAY)
            line_pct = (gray < 200).mean()  # пиксели темнее 200/255

            if line_pct >= self.min_line_pct:
                lq_patch = lq[y:y+ps, x:x+ps]
                break
        else:
            # если не нашли — берём случайный
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
            lq_patch = lq[y:y+ps, x:x+ps]
            hq_patch = hq[y:y+ps, x:x+ps]

        # Аугментации
        if self.augment:
            if random.random() < 0.5:
                lq_patch = np.fliplr(lq_patch).copy()
                hq_patch = np.fliplr(hq_patch).copy()
            if random.random() < 0.5:
                lq_patch = np.flipud(lq_patch).copy()
                hq_patch = np.flipud(hq_patch).copy()
            k = random.randint(0, 3)
            if k > 0:
                lq_patch = np.rot90(lq_patch, k).copy()
                hq_patch = np.rot90(hq_patch, k).copy()

        # Нормализация
        lq_patch = torch.from_numpy(lq_patch.astype(np.float32) / 255.0).permute(2, 0, 1)
        hq_patch = torch.from_numpy(hq_patch.astype(np.float32) / 255.0).permute(2, 0, 1)

        return lq_patch, hq_patch, lq_fname


# =========================================================
# MODEL — с BatchNorm и LeakyReLU
# =========================================================
class ConvBlock(nn.Module):
    """Два Conv2d + BatchNorm + LeakyReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ConvAE(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dropout_rate=0.3):
        super().__init__()

        # ---- ENCODER ----
        self.enc1 = ConvBlock(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        # ---- BOTTLENECK ----
        self.bottleneck_conv = nn.Conv2d(256, 512, 3, padding=1, bias=False)
        self.bottleneck_bn = nn.BatchNorm2d(512)
        self.bottleneck_relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout_rate)

        # ---- DECODER ----
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2, bias=False)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2, bias=False)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2, bias=False)
        self.dec1 = ConvBlock(64 + 64, 64)

        # ---- OUTPUT ----
        self.final = nn.Sequential(
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.bottleneck_conv(p3)
        b = self.bottleneck_bn(b)
        b = self.bottleneck_relu(b)
        b = self.dropout(b)

        # Decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return out


# =========================================================
# INIT WEIGHTS
# =========================================================
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


# =========================================================
# LOSS
# =========================================================
def combined_loss(pred, target):
    """L1 + SSIM с упором на не-белые пиксели (линии, стены)."""

    # Маска: где изображение НЕ белое (фон)
    # Чем темнее пиксель — тем важнее
    gray_target = 0.299 * target[:, 0:1] + 0.587 * target[:, 1:2] + 0.114 * target[:, 2:3]
    weight = 1.0 + 5.0 * (1.0 - gray_target)  # Белый фон → вес ~1, чёрная линия → вес ~6

    # Взвешенный L1
    l1_map = torch.abs(pred - target)
    weighted_l1 = (l1_map * weight).mean()

    # SSIM как обычно
    ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
    ssim_loss = 1.0 - ssim_val

    total = weighted_l1 + 0.5 * ssim_loss
    return total, weighted_l1, ssim_loss, ssim_val


# =========================================================
# PSNR
# =========================================================
def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


# =========================================================
# VISUALISATION
# =========================================================
def tensor_to_numpy(t):
    return np.clip(t.detach().cpu().permute(1, 2, 0).numpy(), 0, 1)


def log_comparison(lq, pred, hq, iteration, series, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(tensor_to_numpy(lq))
    axes[0].set_title("LQ (Input)")
    axes[0].axis("off")
    axes[1].imshow(tensor_to_numpy(pred))
    axes[1].set_title("Prediction")
    axes[1].axis("off")
    axes[2].imshow(tensor_to_numpy(hq))
    axes[2].set_title("HQ (Target)")
    axes[2].axis("off")
    plt.tight_layout()
    logger.report_matplotlib_figure(title=title, series=series, figure=fig, iteration=iteration)
    plt.close(fig)


# =========================================================
# REPRODUCIBILITY
# =========================================================
random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = True


# =========================================================
# TRAIN / TEST SPLIT (80% / 20%)
# =========================================================
full_dataset = FloorPlanDataset(lq_dir, hq_dir, augment=True)

total = len(full_dataset)
test_size = max(1, int(total * config["test_split"]))
train_size = total - test_size

gen = torch.Generator().manual_seed(config["seed"])
train_ds, test_ds = random_split(full_dataset, [train_size, test_size], generator=gen)

train_loader = DataLoader(
    train_ds,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
)

print(f"Total: {total} | Train: {train_size} (80%) | Test: {test_size} (20%)")
logger.report_text(f"Total: {total} | Train: {train_size} | Test: {test_size}")


# =========================================================
# INIT MODEL, OPTIMIZER, SCHEDULER
# =========================================================
model = ConvAE(in_ch=3, out_ch=3, dropout_rate=config["dropout_rate"])
model.apply(init_weights)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
logger.report_text(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

best_test_psnr = 0.0
epochs_no_improve = 0
best_model_path = os.path.join(config["save_dir"], "best_model.pth")


# =========================================================
# ДИАГНОСТИКА ПЕРЕД ОБУЧЕНИЕМ
# =========================================================
print("\n=== Pre-training diagnostics ===")
model.eval()
with torch.no_grad():
    sample_lq, sample_hq, _ = next(iter(train_loader))
    sample_lq = sample_lq.to(device)
    sample_pred = model(sample_lq)

    print(f"LQ    — min: {sample_lq.min().item():.4f}, max: {sample_lq.max().item():.4f}, mean: {sample_lq.mean().item():.4f}")
    print(f"Pred  — min: {sample_pred.min().item():.4f}, max: {sample_pred.max().item():.4f}, mean: {sample_pred.mean().item():.4f}")
    print(f"HQ    — min: {sample_hq.min().item():.4f}, max: {sample_hq.max().item():.4f}, mean: {sample_hq.mean().item():.4f}")
    print(f"Initial PSNR: {calc_psnr(sample_pred[:1], sample_hq[:1].to(device)).item():.2f} dB")
print("===============================\n")


# =========================================================
# TRAINING LOOP
# =========================================================
for epoch in range(config["epochs"]):
    # ===================== TRAIN =====================
    model.train()
    train_loss_sum = 0.0
    train_l1_sum = 0.0
    train_ssim_loss_sum = 0.0

    for lq, hq, _ in train_loader:
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            pred = model(lq)
            loss, l1_part, ssim_loss_part, _ = combined_loss(pred, hq)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()
        train_l1_sum += l1_part.item()
        train_ssim_loss_sum += ssim_loss_part.item()

    n_train = len(train_loader)
    t_loss = train_loss_sum / n_train
    t_l1 = train_l1_sum / n_train
    t_ssim = train_ssim_loss_sum / n_train

    scheduler.step()

    logger.report_scalar("Loss", "train_total", t_loss, epoch)
    logger.report_scalar("Loss", "train_L1", t_l1, epoch)
    logger.report_scalar("Loss", "train_SSIM_loss", t_ssim, epoch)
    logger.report_scalar("LR", "learning_rate", scheduler.get_last_lr()[0], epoch)

    # ===================== TEST =====================
    model.eval()
    test_psnr_sum = 0.0
    test_ssim_sum = 0.0
    test_sample = None

    with torch.no_grad():
        for i, (lq, hq, _) in enumerate(test_loader):
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            pred = model(lq)

            test_psnr_sum += calc_psnr(pred, hq).item()
            test_ssim_sum += ssim(pred, hq, data_range=1.0, size_average=True).item()

            if i == 0:
                test_sample = (lq[0].cpu(), pred[0].cpu(), hq[0].cpu())

    n_test = len(test_loader)
    test_psnr = test_psnr_sum / n_test
    test_ssim = test_ssim_sum / n_test

    logger.report_scalar("PSNR", "test", test_psnr, epoch)
    logger.report_scalar("SSIM", "test", test_ssim, epoch)

    if test_sample is not None:
        log_comparison(
            test_sample[0],
            test_sample[1],
            test_sample[2],
            iteration=epoch,
            series="test_examples",
            title=f"Test — Epoch {epoch+1} | PSNR: {test_psnr:.2f} dB",
        )

    print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
          f"Train Loss: {t_loss:.4f} | "
          f"Test PSNR: {test_psnr:.2f} dB | "
          f"Test SSIM: {test_ssim:.4f}")

    # ===================== SAVE BEST / EARLY STOPPING =====================
    if test_psnr > best_test_psnr:
        best_test_psnr = test_psnr
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_model_path)
        task.upload_artifact("best_model", best_model_path)
        print(f"  >> New best model! PSNR: {best_test_psnr:.2f} dB")
    else:
        epochs_no_improve += 1
        print(f"  No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= config["early_stopping_patience"]:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # ===================== SAVE CHECKPOINT =====================
    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(config["save_dir"], f"checkpoint_epoch_{epoch+1:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)
        task.upload_artifact(f"checkpoint_epoch_{epoch+1:03d}", ckpt_path)


# =========================================================
# FINAL SUMMARY
# =========================================================
summary = {
    "best_test_psnr_dB": best_test_psnr,
    "target_psnr_dB": 30.0,
    "target_reached": best_test_psnr >= 30.0,
    "model_architecture": "ConvAE: Encoder 3→64→128→256, MaxPoolx3, Bottleneck 256→512+Dropout, Decoder ConvTranspose2D+Skip, Sigmoid",
    "latent_shape": "[B, 512, 125, 125]",
    "train_test_split": f"{train_size}/{test_size} (80/20)",
    "improvements": "BatchNorm + LeakyReLU + Kaiming init + lr=5e-5",
    "config": config,
}

summary_path = os.path.join(config["save_dir"], "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

task.upload_artifact("summary", summary_path)
task.upload_artifact("outputs_folder", config["save_dir"])

print(f"\n{'='*60}")
print(f"Training complete!")
print(f"Best Test PSNR: {best_test_psnr:.2f} dB")
print(f"Target PSNR:   30.00 dB {'✅' if best_test_psnr >= 30.0 else '❌'}")
print(f"Outputs: {config['save_dir']}")
print(f"{'='*60}")

task.close()

