import os
import cv2
import math
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from clearml import Task, Dataset as ClearMLDataset
from pytorch_msssim import ssim


# =========================
# 1. CLEARML CONNECTION
# =========================

Task.set_credentials(
    api_host="https://api.clear.ml",
    web_host="https://app.clear.ml",
    files_host="https://files.clear.ml",
    key="DO2Z4JRJIERU61YCI6II5AD1JZ5M32",
    secret="3T44b4ejaO0PeO6ePDgLW-72sUYw85HgAtZnGUvxr69mUa5WuGWLkKrWpaMxWoHbM2U"
)

task = Task.init(
    project_name="Vosstanovlenie_tehnicheskih_sistem",
    task_name="Train UNet ConvAE",
    task_type=Task.TaskTypes.training
)

logger = task.get_logger()


# =========================
# 2. CONFIG
# =========================

config = {
    "clearml_dataset_id": "4ffc5ed351114bd79588dc87ec15b2c7",
    "patch_size": 256,
    "batch_size": 8,
    "epochs": 50,
    "lr": 3e-4,
    "val_ratio": 0.1,
    "num_workers": 2,
    "base_channels": 32,
    "tile_size": 256,
    "tile_overlap": 32,
    "save_dir": "outputs"
}

config = task.connect(config)

os.makedirs(config["save_dir"], exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# =========================
# 3. DOWNLOAD DATASET FROM CLEARML
# =========================

dataset = ClearMLDataset.get(dataset_id=config["clearml_dataset_id"])
local_dataset_path = dataset.get_local_copy()

print("ClearML dataset downloaded to:", local_dataset_path)

# Ожидаем структуру:
# local_dataset_path/
#   train/
#     LQ/
#     HQ/
#   test/
#     LQ/
#     HQ/

train_lq_dir = os.path.join(local_dataset_path, "train", "LQ")
train_hq_dir = os.path.join(local_dataset_path, "train", "HQ")
test_lq_dir  = os.path.join(local_dataset_path, "test", "LQ")
test_hq_dir  = os.path.join(local_dataset_path, "test", "HQ")

assert os.path.isdir(train_lq_dir), f"Not found: {train_lq_dir}"
assert os.path.isdir(train_hq_dir), f"Not found: {train_hq_dir}"
assert os.path.isdir(test_lq_dir), f"Not found: {test_lq_dir}"
assert os.path.isdir(test_hq_dir), f"Not found: {test_hq_dir}"


# =========================
# 4. DATASET
# =========================

def list_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


class LQHQPatchDataset(Dataset):
    def __init__(self, lq_dir, hq_dir, patch_size=256):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.patch_size = patch_size

        self.names = list_image_files(lq_dir)
        assert len(self.names) > 0, f"No images found in {lq_dir}"

        missing = [n for n in self.names if not os.path.exists(os.path.join(hq_dir, n))]
        assert len(missing) == 0, f"Missing matching HQ files for: {missing[:5]}"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        lq = cv2.imread(os.path.join(self.lq_dir, name), cv2.IMREAD_GRAYSCALE)
        hq = cv2.imread(os.path.join(self.hq_dir, name), cv2.IMREAD_GRAYSCALE)

        if lq is None:
            raise ValueError(f"Failed to read LQ image: {name}")
        if hq is None:
            raise ValueError(f"Failed to read HQ image: {name}")

        if lq.shape != hq.shape:
            raise ValueError(f"Shape mismatch for {name}: LQ={lq.shape}, HQ={hq.shape}")

        h, w = lq.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(f"Image {name} smaller than patch size {ps}: got {lq.shape}")

        x = random.randint(0, w - ps)
        y = random.randint(0, h - ps)

        lq = lq[y:y+ps, x:x+ps]
        hq = hq[y:y+ps, x:x+ps]

        lq = lq.astype(np.float32) / 255.0
        hq = hq.astype(np.float32) / 255.0

        lq = torch.from_numpy(lq).unsqueeze(0)
        hq = torch.from_numpy(hq).unsqueeze(0)

        return lq, hq, name


class LQHQFullImageDataset(Dataset):
    def __init__(self, lq_dir, hq_dir):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.names = list_image_files(lq_dir)
        assert len(self.names) > 0, f"No images found in {lq_dir}"

        missing = [n for n in self.names if not os.path.exists(os.path.join(hq_dir, n))]
        assert len(missing) == 0, f"Missing matching HQ files for: {missing[:5]}"

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]

        lq = cv2.imread(os.path.join(self.lq_dir, name), cv2.IMREAD_GRAYSCALE)
        hq = cv2.imread(os.path.join(self.hq_dir, name), cv2.IMREAD_GRAYSCALE)

        if lq is None or hq is None:
            raise ValueError(f"Failed to read full image pair for {name}")

        lq = lq.astype(np.float32) / 255.0
        hq = hq.astype(np.float32) / 255.0

        lq = torch.from_numpy(lq).unsqueeze(0)
        hq = torch.from_numpy(hq).unsqueeze(0)

        return lq, hq, name


# =========================
# 5. MODEL
# =========================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetConvAE(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()

        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)       # 256
        self.enc2 = DownBlock(b, b * 2)             # 128
        self.enc3 = DownBlock(b * 2, b * 4)         # 64
        self.enc4 = DownBlock(b * 4, b * 8)         # 32

        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 16, kernel_size=3, stride=2, padding=1),  # 16
            nn.ReLU(inplace=True),
            nn.Conv2d(b * 16, b * 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up4 = UpBlock(b * 16, b * 8, b * 8)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)

        self.final = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        out = torch.sigmoid(self.final(d1))
        return out


# =========================
# 6. METRICS / HELPERS
# =========================

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def save_comparison_image(lq, pred, hq, save_path, title=None):
    lq_np = lq.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()
    hq_np = hq.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(lq_np, cmap="gray")
    axes[0].set_title("LQ")
    axes[1].imshow(pred_np, cmap="gray")
    axes[1].set_title("Output")
    axes[2].imshow(hq_np, cmap="gray")
    axes[2].set_title("HQ")

    if title:
        fig.suptitle(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def report_comparison_to_clearml(lq, pred, hq, iteration, series="validation_examples"):
    lq_np = lq.squeeze().detach().cpu().numpy()
    pred_np = pred.squeeze().detach().cpu().numpy()
    hq_np = hq.squeeze().detach().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(lq_np, cmap="gray")
    axes[0].set_title("LQ")
    axes[1].imshow(pred_np, cmap="gray")
    axes[1].set_title("Output")
    axes[2].imshow(hq_np, cmap="gray")
    axes[2].set_title("HQ")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    logger.report_matplotlib_figure(
        title="LQ_vs_Output_vs_HQ",
        series=series,
        figure=fig,
        iteration=iteration
    )
    plt.close(fig)


# =========================
# 7. TILED INFERENCE
# =========================

def tiled_inference(model, image_tensor, tile_size=256, overlap=32, device="cuda"):
    """
    image_tensor: [1, H, W]
    returns: [1, H, W]
    """
    model.eval()

    _, H, W = image_tensor.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be > overlap")

    output = torch.zeros((1, H, W), dtype=torch.float32)
    weight = torch.zeros((1, H, W), dtype=torch.float32)

    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))

    if len(ys) == 0 or ys[-1] != H - tile_size:
        ys.append(max(H - tile_size, 0))
    if len(xs) == 0 or xs[-1] != W - tile_size:
        xs.append(max(W - tile_size, 0))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image_tensor[:, y:y+tile_size, x:x+tile_size]
                if patch.shape[-2:] != (tile_size, tile_size):
                    pad_h = tile_size - patch.shape[-2]
                    pad_w = tile_size - patch.shape[-1]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

                pred = model(patch.unsqueeze(0).to(device)).cpu().squeeze(0)

                pred = pred[:, :min(tile_size, H - y), :min(tile_size, W - x)]

                output[:, y:y+pred.shape[1], x:x+pred.shape[2]] += pred
                weight[:, y:y+pred.shape[1], x:x+pred.shape[2]] += 1.0

    output = output / torch.clamp(weight, min=1e-8)
    return output


# =========================
# 8. DATALOADERS
# =========================

full_train_dataset = LQHQPatchDataset(
    lq_dir=train_lq_dir,
    hq_dir=train_hq_dir,
    patch_size=config["patch_size"]
)

val_size = max(1, int(len(full_train_dataset) * config["val_ratio"]))
train_size = len(full_train_dataset) - val_size

train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

test_dataset = LQHQFullImageDataset(
    lq_dir=test_lq_dir,
    hq_dir=test_hq_dir
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config["batch_size"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True
)


# =========================
# 9. TRAIN SETUP
# =========================

model = UNetConvAE(
    in_channels=1,
    out_channels=1,
    base_channels=config["base_channels"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

best_val_ssim = -1.0
best_model_path = os.path.join(config["save_dir"], "best_model.pth")


# =========================
# 10. TRAIN LOOP
# =========================

for epoch in range(config["epochs"]):
    model.train()
    running_train_loss = 0.0

    for lq, hq, _ in train_loader:
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        pred = model(lq)
        loss = F.l1_loss(pred, hq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    avg_train_loss = running_train_loss / max(len(train_loader), 1)

    model.eval()
    running_val_loss = 0.0
    running_val_psnr = 0.0
    running_val_ssim = 0.0

    val_example_logged = False

    with torch.no_grad():
        for lq, hq, names in val_loader:
            lq = lq.to(device, non_blocking=True)
            hq = hq.to(device, non_blocking=True)

            pred = model(lq)

            val_loss = F.l1_loss(pred, hq)
            val_psnr = calc_psnr(pred, hq)
            val_ssim = ssim(pred, hq, data_range=1.0, size_average=True)

            running_val_loss += val_loss.item()
            running_val_psnr += val_psnr.item()
            running_val_ssim += val_ssim.item()

            if not val_example_logged:
                comparison_path = os.path.join(config["save_dir"], f"val_epoch_{epoch:03d}.png")
                save_comparison_image(
                    lq[0], pred[0], hq[0],
                    save_path=comparison_path,
                    title=f"Validation epoch {epoch}"
                )
                report_comparison_to_clearml(lq[0], pred[0], hq[0], iteration=epoch)
                val_example_logged = True

    avg_val_loss = running_val_loss / max(len(val_loader), 1)
    avg_val_psnr = running_val_psnr / max(len(val_loader), 1)
    avg_val_ssim = running_val_ssim / max(len(val_loader), 1)

    print(
        f"Epoch {epoch+1}/{config['epochs']} | "
        f"train_loss={avg_train_loss:.6f} | "
        f"val_loss={avg_val_loss:.6f} | "
        f"val_psnr={avg_val_psnr:.4f} | "
        f"val_ssim={avg_val_ssim:.4f}"
    )

    logger.report_scalar("loss", "train", avg_train_loss, epoch)
    logger.report_scalar("loss", "val", avg_val_loss, epoch)
    logger.report_scalar("psnr", "val", avg_val_psnr, epoch)
    logger.report_scalar("ssim", "val", avg_val_ssim, epoch)

    if avg_val_ssim > best_val_ssim:
        best_val_ssim = avg_val_ssim
        torch.save(model.state_dict(), best_model_path)
        task.upload_artifact("best_model.pth", best_model_path)


# =========================
# 11. LOAD BEST MODEL
# =========================

print("Best val SSIM:", best_val_ssim)
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()


# =========================
# 12. TEST + TILED INFERENCE 1000x1000
# =========================

test_psnr_sum = 0.0
test_ssim_sum = 0.0
test_count = 0

for i, (lq, hq, names) in enumerate(test_loader):
    name = names[0]

    lq = lq[0]   # [1, H, W]
    hq = hq[0]   # [1, H, W]

    pred = tiled_inference(
        model=model,
        image_tensor=lq,
        tile_size=config["tile_size"],
        overlap=config["tile_overlap"],
        device=device
    )

    pred_batch = pred.unsqueeze(0)
    hq_batch = hq.unsqueeze(0)

    image_psnr = calc_psnr(pred_batch, hq_batch).item()
    image_ssim = ssim(pred_batch, hq_batch, data_range=1.0, size_average=True).item()

    test_psnr_sum += image_psnr
    test_ssim_sum += image_ssim
    test_count += 1

    save_path = os.path.join(config["save_dir"], f"test_compare_{i:03d}_{name}.png")
    save_comparison_image(
        lq, pred, hq,
        save_path=save_path,
        title=f"{name} | PSNR={image_psnr:.3f}, SSIM={image_ssim:.4f}"
    )

    report_comparison_to_clearml(
        lq, pred, hq,
        iteration=i,
        series="test_examples"
    )

avg_test_psnr = test_psnr_sum / max(test_count, 1)
avg_test_ssim = test_ssim_sum / max(test_count, 1)

logger.report_scalar("psnr", "test", avg_test_psnr, 0)
logger.report_scalar("ssim", "test", avg_test_ssim, 0)

print(f"Test PSNR: {avg_test_psnr:.4f}")
print(f"Test SSIM: {avg_test_ssim:.4f}")


# =========================
# 13. SAVE SUMMARY
# =========================

summary = {
    "best_val_ssim": best_val_ssim,
    "avg_test_psnr": avg_test_psnr,
    "avg_test_ssim": avg_test_ssim,
    "config": config
}

summary_path = os.path.join(config["save_dir"], "summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

task.upload_artifact("summary.json", summary_path)

# Можно загрузить всю папку outputs как артефакт
task.upload_artifact("outputs_folder", artifact_object=config["save_dir"])

print("Training finished. Outputs saved in:", config["save_dir"])
