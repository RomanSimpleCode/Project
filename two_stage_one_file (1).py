
import os
import cv2
import json
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from clearml import Task, Dataset as ClearMLDataset
from pytorch_msssim import ssim


# =========================================================
# 1. CLEARML TASK
# =========================================================

task = Task.init(
    project_name="Vosstanovlenie_tehnicheskih_sistem",
    task_name="Train Residual UNet ConvAE RGB TwoStage",
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False,
)
logger = task.get_logger()


# =========================================================
# 2. CONFIG
# =========================================================
# stage1 = стабильная база
# stage2 = аккуратный fine-tune от best checkpoint на резкость

config = {
    "clearml_dataset_id": "3ea1e9f808034406bdf383ff1bbb32f4",

    # one of: "stage1", "stage2"
    "mode": "stage1",

    # for stage2: path to checkpoint from stage1
    "resume_checkpoint": "",

    # data
    "patch_size": 384,
    "tile_size": 512,
    "tile_overlap": 64,
    "test_ratio": 0.2,
    "test_subset_size": 8,
    "num_workers": 2,

    # crop
    "min_content_ratio": 0.18,
    "max_crop_attempts": 80,

    # train
    "batch_size": 2,
    "epochs_stage1": 40,
    "epochs_stage2": 20,
    "lr_stage1": 5e-5,
    "lr_stage2": 1e-5,
    "base_channels": 16,

    # logging
    "eval_every": 1,
    "save_every": 5,

    # stage1 loss: stable
    "stage1_l1_weight": 1.0,
    "stage1_ssim_weight": 0.0,
    "stage1_edge_weight": 0.0,
    "stage1_content_boost": 2.0,
    "stage1_residual_scale": 0.7,

    # stage2 loss: sharpen slightly
    "stage2_l1_weight": 0.95,
    "stage2_ssim_weight": 0.0,
    "stage2_edge_weight": 0.05,
    "stage2_content_boost": 2.0,
    "stage2_residual_scale": 0.7,

    # misc
    "save_dir": "outputs",
    "seed": 42,
}

config = task.connect(config)
os.makedirs(config["save_dir"], exist_ok=True)


# =========================================================
# 3. DEVICE CHECK
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    raise RuntimeError(
        "GPU is not enabled. In Colab: Runtime -> Change runtime type -> GPU. "
        "Do not start training on CPU."
    )


# =========================================================
# 4. DOWNLOAD DATASET FROM CLEARML
# =========================================================

dataset = ClearMLDataset.get(dataset_id=config["clearml_dataset_id"])
local_dataset_path = dataset.get_local_copy()

print("ClearML dataset downloaded to:", local_dataset_path)
print("Dataset root content:", os.listdir(local_dataset_path))

lq_dir = os.path.join(local_dataset_path, "LQ")
hq_dir = os.path.join(local_dataset_path, "HQ")

assert os.path.isdir(lq_dir), f"Not found: {lq_dir}"
assert os.path.isdir(hq_dir), f"Not found: {hq_dir}"


# =========================================================
# 5. DATASET
# =========================================================

def list_image_files(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted([f for f in os.listdir(folder) if f.lower().endswith(exts)])


def estimate_content_ratio(rgb_patch: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_patch, cv2.COLOR_RGB2GRAY)
    mask = gray < 245
    return float(mask.mean())


class LQHQPatchDataset(Dataset):
    def __init__(
        self,
        lq_dir,
        hq_dir,
        patch_size=384,
        min_content_ratio=0.18,
        max_crop_attempts=80,
    ):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir
        self.patch_size = patch_size
        self.min_content_ratio = min_content_ratio
        self.max_crop_attempts = max_crop_attempts

        self.lq_names = list_image_files(lq_dir)
        self.hq_names = list_image_files(hq_dir)

        assert len(self.lq_names) > 0, f"No images found in {lq_dir}"
        assert len(self.lq_names) == len(self.hq_names), "LQ/HQ file count mismatch"

        lq_stems = [os.path.splitext(n)[0] for n in self.lq_names]
        hq_stems = [os.path.splitext(n)[0] for n in self.hq_names]
        assert lq_stems == hq_stems, "LQ/HQ stem mismatch"

        self.names = self.lq_names
        self.hq_map = {os.path.splitext(n)[0]: n for n in self.hq_names}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        lq_name = self.names[idx]
        stem = os.path.splitext(lq_name)[0]
        hq_name = self.hq_map[stem]

        lq = cv2.imread(os.path.join(self.lq_dir, lq_name), cv2.IMREAD_COLOR)
        hq = cv2.imread(os.path.join(self.hq_dir, hq_name), cv2.IMREAD_COLOR)

        if lq is None:
            raise ValueError(f"Failed to read LQ image: {lq_name}")
        if hq is None:
            raise ValueError(f"Failed to read HQ image: {hq_name}")

        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        hq = cv2.cvtColor(hq, cv2.COLOR_BGR2RGB)

        if lq.shape != hq.shape:
            raise ValueError(f"Shape mismatch for {stem}: LQ={lq.shape}, HQ={hq.shape}")

        h, w, _ = lq.shape
        ps = self.patch_size

        if h < ps or w < ps:
            raise ValueError(f"Image {stem} smaller than patch size {ps}: got {lq.shape}")

        best_patch = None
        best_score = -1.0

        for _ in range(self.max_crop_attempts):
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)

            lq_patch = lq[y:y+ps, x:x+ps]
            hq_patch = hq[y:y+ps, x:x+ps]

            score = estimate_content_ratio(hq_patch)

            if score > best_score:
                best_score = score
                best_patch = (lq_patch, hq_patch)

            if score >= self.min_content_ratio:
                best_patch = (lq_patch, hq_patch)
                break

        lq_patch, hq_patch = best_patch

        lq_patch = lq_patch.astype(np.float32) / 255.0
        hq_patch = hq_patch.astype(np.float32) / 255.0

        lq_patch = torch.from_numpy(lq_patch).permute(2, 0, 1)
        hq_patch = torch.from_numpy(hq_patch).permute(2, 0, 1)

        return lq_patch, hq_patch, stem


class LQHQFullImageDataset(Dataset):
    def __init__(self, lq_dir, hq_dir):
        self.lq_dir = lq_dir
        self.hq_dir = hq_dir

        self.lq_names = list_image_files(lq_dir)
        self.hq_names = list_image_files(hq_dir)

        assert len(self.lq_names) > 0, f"No images found in {lq_dir}"
        assert len(self.lq_names) == len(self.hq_names), "LQ/HQ file count mismatch"

        lq_stems = [os.path.splitext(n)[0] for n in self.lq_names]
        hq_stems = [os.path.splitext(n)[0] for n in self.hq_names]
        assert lq_stems == hq_stems, "LQ/HQ stem mismatch"

        self.names = self.lq_names
        self.hq_map = {os.path.splitext(n)[0]: n for n in self.hq_names}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        lq_name = self.names[idx]
        stem = os.path.splitext(lq_name)[0]
        hq_name = self.hq_map[stem]

        lq = cv2.imread(os.path.join(self.lq_dir, lq_name), cv2.IMREAD_COLOR)
        hq = cv2.imread(os.path.join(self.hq_dir, hq_name), cv2.IMREAD_COLOR)

        if lq is None or hq is None:
            raise ValueError(f"Failed to read full image pair for {stem}")

        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        hq = cv2.cvtColor(hq, cv2.COLOR_BGR2RGB)

        lq = lq.astype(np.float32) / 255.0
        hq = hq.astype(np.float32) / 255.0

        lq = torch.from_numpy(lq).permute(2, 0, 1)
        hq = torch.from_numpy(hq).permute(2, 0, 1)

        return lq, hq, stem


# =========================================================
# 6. MODEL
# =========================================================

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetConvAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=16):
        super().__init__()

        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(b * 8, b * 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(b * 16, b * 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
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

        return self.final(d1)


# =========================================================
# 7. HELPERS
# =========================================================

def calc_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10(1.0 / mse)


def rgb_to_gray(x):
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]


def make_sobel_kernels(device, dtype):
    sobel_x = torch.tensor(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]],
        device=device,
        dtype=dtype
    ).view(1, 1, 3, 3)

    return sobel_x, sobel_y


def apply_residual(lq, residual_raw, residual_scale):
    pred = lq + residual_scale * torch.tanh(residual_raw)
    pred = torch.clamp(pred, 0.0, 1.0)
    return pred


def reconstruction_loss(pred, target, l1_weight, ssim_weight, edge_weight, content_boost):
    target = torch.clamp(target, 0.0, 1.0)
    pred = torch.clamp(pred, 0.0, 1.0)

    target_gray = rgb_to_gray(target)
    pred_gray = rgb_to_gray(pred)

    content_mask = 1.0 - target_gray
    pixel_weight = 1.0 + content_boost * content_mask

    l1_map = torch.abs(pred - target)
    weighted_l1 = (l1_map * pixel_weight).mean()

    if ssim_weight > 0.0:
        ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
    else:
        ssim_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

    if edge_weight > 0.0:
        sobel_x, sobel_y = make_sobel_kernels(pred.device, pred.dtype)
        pred_gx = F.conv2d(pred_gray, sobel_x, padding=1)
        pred_gy = F.conv2d(pred_gray, sobel_y, padding=1)
        targ_gx = F.conv2d(target_gray, sobel_x, padding=1)
        targ_gy = F.conv2d(target_gray, sobel_y, padding=1)

        pred_edges = torch.sqrt(pred_gx ** 2 + pred_gy ** 2 + 1e-6)
        targ_edges = torch.sqrt(targ_gx ** 2 + targ_gy ** 2 + 1e-6)
        edge_loss = F.l1_loss(pred_edges, targ_edges)
    else:
        edge_loss = torch.zeros((), device=pred.device, dtype=pred.dtype)

    total = l1_weight * weighted_l1 + ssim_weight * ssim_loss + edge_weight * edge_loss
    return total, weighted_l1, ssim_loss, edge_loss


def chw_to_hwc(x):
    return np.transpose(x.detach().cpu().numpy(), (1, 2, 0))


def save_comparison_image(lq, pred, hq, save_path, title=None):
    lq_np = np.clip(chw_to_hwc(lq), 0, 1)
    pred_np = np.clip(chw_to_hwc(pred), 0, 1)
    hq_np = np.clip(chw_to_hwc(hq), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(lq_np)
    axes[0].set_title("LQ")
    axes[1].imshow(pred_np)
    axes[1].set_title("Output")
    axes[2].imshow(hq_np)
    axes[2].set_title("HQ")

    if title:
        fig.suptitle(title)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def report_comparison_to_clearml(lq, pred, hq, iteration, series, title):
    lq_np = np.clip(chw_to_hwc(lq), 0, 1)
    pred_np = np.clip(chw_to_hwc(pred), 0, 1)
    hq_np = np.clip(chw_to_hwc(hq), 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(lq_np)
    axes[0].set_title("LQ")
    axes[1].imshow(pred_np)
    axes[1].set_title("Output")
    axes[2].imshow(hq_np)
    axes[2].set_title("HQ")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    logger.report_matplotlib_figure(
        title=title,
        series=series,
        figure=fig,
        iteration=iteration
    )
    plt.close(fig)


def tiled_inference_residual(model, image_tensor, tile_size, overlap, device):
    model.eval()

    c, H, W = image_tensor.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be > overlap")

    output = torch.zeros((c, H, W), dtype=torch.float32)
    weight = torch.zeros((c, H, W), dtype=torch.float32)

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

                residual = model(patch.unsqueeze(0).to(device)).cpu().squeeze(0)
                residual = residual[:, :min(tile_size, H - y), :min(tile_size, W - x)]

                output[:, y:y+residual.shape[1], x:x+residual.shape[2]] += residual
                weight[:, y:y+residual.shape[1], x:x+residual.shape[2]] += 1.0

    return output / torch.clamp(weight, min=1e-8)


def get_stage_params(cfg):
    mode = cfg["mode"].strip().lower()
    if mode == "stage1":
        return {
            "epochs": cfg["epochs_stage1"],
            "lr": cfg["lr_stage1"],
            "l1_weight": cfg["stage1_l1_weight"],
            "ssim_weight": cfg["stage1_ssim_weight"],
            "edge_weight": cfg["stage1_edge_weight"],
            "content_boost": cfg["stage1_content_boost"],
            "residual_scale": cfg["stage1_residual_scale"],
        }
    elif mode == "stage2":
        return {
            "epochs": cfg["epochs_stage2"],
            "lr": cfg["lr_stage2"],
            "l1_weight": cfg["stage2_l1_weight"],
            "ssim_weight": cfg["stage2_ssim_weight"],
            "edge_weight": cfg["stage2_edge_weight"],
            "content_boost": cfg["stage2_content_boost"],
            "residual_scale": cfg["stage2_residual_scale"],
        }
    else:
        raise ValueError("config['mode'] must be 'stage1' or 'stage2'")


# =========================================================
# 8. REPRODUCIBILITY
# =========================================================

random.seed(config["seed"])
np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
torch.cuda.manual_seed_all(config["seed"])
torch.backends.cudnn.benchmark = True


# =========================================================
# 9. DATALOADERS
# =========================================================

full_patch_dataset = LQHQPatchDataset(
    lq_dir=lq_dir,
    hq_dir=hq_dir,
    patch_size=config["patch_size"],
    min_content_ratio=config["min_content_ratio"],
    max_crop_attempts=config["max_crop_attempts"],
)

full_image_dataset = LQHQFullImageDataset(lq_dir=lq_dir, hq_dir=hq_dir)

total_size = len(full_patch_dataset)
test_size = max(1, int(total_size * config["test_ratio"]))
train_size = total_size - test_size

if train_size <= 0:
    raise ValueError("test_ratio produced empty train split. Decrease test_ratio.")

split_generator = torch.Generator().manual_seed(config["seed"])
indices = list(range(total_size))
train_indices, test_indices = random_split(
    indices,
    [train_size, test_size],
    generator=split_generator
)

train_dataset = Subset(full_patch_dataset, train_indices.indices)
val_indices = test_indices.indices[:config["test_subset_size"]]
test_image_subset = Subset(full_image_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True,
    persistent_workers=(config["num_workers"] > 0),
)

test_loader = DataLoader(
    test_image_subset,
    batch_size=1,
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
    persistent_workers=(config["num_workers"] > 0),
)

print(f"Full dataset: {total_size}")
print(f"Train patches dataset: {len(train_dataset)}")
print(f"Validation full-images used: {len(test_image_subset)}")


# =========================================================
# 10. TRAIN SETUP
# =========================================================

stage = get_stage_params(config)
print("Stage params:", stage)

model = UNetConvAE(
    in_channels=3,
    out_channels=3,
    base_channels=config["base_channels"]
).to(device)

if config["mode"].lower() == "stage2":
    if not config["resume_checkpoint"]:
        raise ValueError("For stage2, set config['resume_checkpoint'] to best_model.pth from stage1")
    print("Loading checkpoint:", config["resume_checkpoint"])
    state_dict = torch.load(config["resume_checkpoint"], map_location=device)
    model.load_state_dict(state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=stage["lr"])
scaler = torch.amp.GradScaler("cuda", enabled=True)

best_val_ssim = -1.0
best_model_name = f"best_model_{config['mode'].lower()}.pth"
best_model_path = os.path.join(config["save_dir"], best_model_name)


# =========================================================
# 11. TRAIN LOOP
# =========================================================

for epoch in range(stage["epochs"]):
    model.train()

    running_train_loss = 0.0
    running_train_l1 = 0.0
    running_train_ssim_loss = 0.0
    running_train_edge_loss = 0.0

    train_example_lq = None
    train_example_pred = None
    train_example_hq = None

    for lq, hq, _ in train_loader:
        lq = lq.to(device, non_blocking=True)
        hq = hq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=True):
            residual_raw = model(lq)
            pred = apply_residual(
                lq=lq,
                residual_raw=residual_raw,
                residual_scale=stage["residual_scale"],
            )

            loss, l1_part, ssim_part, edge_part = reconstruction_loss(
                pred=pred,
                target=hq,
                l1_weight=stage["l1_weight"],
                ssim_weight=stage["ssim_weight"],
                edge_weight=stage["edge_weight"],
                content_boost=stage["content_boost"],
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_train_loss += loss.item()
        running_train_l1 += l1_part.item()
        running_train_ssim_loss += ssim_part.item()
        running_train_edge_loss += edge_part.item()

        if train_example_lq is None:
            train_example_lq = lq[0].detach().cpu()
            train_example_pred = pred[0].detach().cpu()
            train_example_hq = hq[0].detach().cpu()

    avg_train_loss = running_train_loss / max(len(train_loader), 1)
    avg_train_l1 = running_train_l1 / max(len(train_loader), 1)
    avg_train_ssim_loss = running_train_ssim_loss / max(len(train_loader), 1)
    avg_train_edge_loss = running_train_edge_loss / max(len(train_loader), 1)

    print(
        f"Epoch {epoch+1}/{stage['epochs']} | "
        f"train_loss={avg_train_loss:.6f} | "
        f"train_l1={avg_train_l1:.6f} | "
        f"train_ssim_loss={avg_train_ssim_loss:.6f} | "
        f"train_edge_loss={avg_train_edge_loss:.6f}"
    )

    logger.report_scalar("loss", f"{config['mode']}_train_total", avg_train_loss, epoch)
    logger.report_scalar("loss", f"{config['mode']}_train_l1", avg_train_l1, epoch)
    logger.report_scalar("loss", f"{config['mode']}_train_ssim_component", avg_train_ssim_loss, epoch)
    logger.report_scalar("loss", f"{config['mode']}_train_edge_component", avg_train_edge_loss, epoch)

    if train_example_lq is not None:
        report_comparison_to_clearml(
            train_example_lq,
            train_example_pred,
            train_example_hq,
            iteration=epoch + 1,
            series=f"{config['mode']}_train_examples",
            title=f"{config['mode']} train comparison epoch {epoch+1}",
        )

    if (epoch + 1) % config["save_every"] == 0:
        checkpoint_path = os.path.join(
            config["save_dir"],
            f"{config['mode']}_checkpoint_epoch_{epoch+1:03d}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        task.upload_artifact(os.path.basename(checkpoint_path), checkpoint_path)

    if (epoch + 1) % config["eval_every"] != 0:
        continue

    model.eval()
    running_val_psnr = 0.0
    running_val_ssim = 0.0
    val_count = 0
    val_example_logged = False

    with torch.no_grad():
        for i, (lq, hq, names) in enumerate(test_loader):
            lq = lq[0].to(device, non_blocking=True)
            hq = hq[0].to(device, non_blocking=True)

            residual_raw = tiled_inference_residual(
                model=model,
                image_tensor=lq,
                tile_size=config["tile_size"],
                overlap=config["tile_overlap"],
                device=device,
            ).to(device)

            pred = apply_residual(
                lq=lq,
                residual_raw=residual_raw,
                residual_scale=stage["residual_scale"],
            )

            pred_batch = pred.unsqueeze(0)
            hq_batch = hq.unsqueeze(0)

            image_psnr = calc_psnr(pred_batch, hq_batch).item()
            image_ssim = ssim(pred_batch, hq_batch, data_range=1.0, size_average=True).item()

            pred_mean = pred_batch.mean().item()
            pred_gray = rgb_to_gray(pred_batch)
            pred_content_ratio = (pred_gray < 0.95).float().mean().item()

            logger.report_scalar("debug", f"{config['mode']}_pred_mean", pred_mean, epoch)
            logger.report_scalar("debug", f"{config['mode']}_pred_content_ratio", pred_content_ratio, epoch)
            logger.report_scalar("debug_rgb", f"{config['mode']}_pred_r_mean", pred_batch[:, 0:1].mean().item(), epoch)
            logger.report_scalar("debug_rgb", f"{config['mode']}_pred_g_mean", pred_batch[:, 1:2].mean().item(), epoch)
            logger.report_scalar("debug_rgb", f"{config['mode']}_pred_b_mean", pred_batch[:, 2:3].mean().item(), epoch)

            running_val_psnr += image_psnr
            running_val_ssim += image_ssim
            val_count += 1

            if not val_example_logged:
                comparison_path = os.path.join(
                    config["save_dir"],
                    f"{config['mode']}_val_epoch_{epoch+1:03d}.png"
                )

                save_comparison_image(
                    lq.cpu(),
                    pred.cpu(),
                    hq.cpu(),
                    save_path=comparison_path,
                    title=f"{config['mode']} validation epoch {epoch+1}",
                )

                report_comparison_to_clearml(
                    lq.cpu(),
                    pred.cpu(),
                    hq.cpu(),
                    iteration=epoch + 1,
                    series=f"{config['mode']}_val_examples",
                    title=f"{config['mode']} validation comparison epoch {epoch+1}",
                )

                task.upload_artifact(
                    name=os.path.basename(comparison_path),
                    artifact_object=comparison_path,
                )

                val_example_logged = True

    avg_val_psnr = running_val_psnr / max(val_count, 1)
    avg_val_ssim = running_val_ssim / max(val_count, 1)

    print(
        f"Epoch {epoch+1}/{stage['epochs']} | "
        f"val_psnr={avg_val_psnr:.4f} | "
        f"val_ssim={avg_val_ssim:.4f}"
    )

    logger.report_scalar("psnr", f"{config['mode']}_val", avg_val_psnr, epoch)
    logger.report_scalar("ssim", f"{config['mode']}_val", avg_val_ssim, epoch)

    if avg_val_ssim > best_val_ssim:
        best_val_ssim = avg_val_ssim
        torch.save(model.state_dict(), best_model_path)
        task.upload_artifact(best_model_name, best_model_path)


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

print("Best validation SSIM:", best_val_ssim)

summary = {
    "mode": config["mode"],
    "best_val_ssim": best_val_ssim,
    "stage": stage,
    "config": config,
}

summary_path = os.path.join(config["save_dir"], f"summary_{config['mode']}.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

task.upload_artifact(os.path.basename(summary_path), summary_path)
task.upload_artifact("outputs_folder", artifact_object=config["save_dir"])

print("Training finished. Outputs saved in:", config["save_dir"])
task.close()
