"""
Streamlit приложение для восстановления изображений через ClearML модель.
Модель: two_stage_one_file.py (UNetConvAE, residual learning)
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from clearml import Task
from PIL import Image

# ------------------------------------------------------------
# Windows guard для ClearML
# ------------------------------------------------------------
if sys.platform == "win32":
    Task.ignore_requirements("pywin32")

# ------------------------------------------------------------
# Константы
# ------------------------------------------------------------
MODEL_TASK_ID = "42a49340fb854bbcaa5424ed068395be"
MODEL_ARTIFACT_NAME = "best_model_stage1.pth"
TARGET_SIZE = 1000  # модель обучалась на патчах/тайлах, приводим к этому размеру
TILE_SIZE = 512
TILE_OVERLAP = 64
RESIDUAL_SCALE = 0.7  # из config stage1


# ------------------------------------------------------------
# Модель (копия из two_stage_one_file.py)
# ------------------------------------------------------------
class ConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.down(x)


class UpBlock(torch.nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetConvAE(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=16):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(in_channels, b)
        self.enc2 = DownBlock(b, b * 2)
        self.enc3 = DownBlock(b * 2, b * 4)
        self.enc4 = DownBlock(b * 4, b * 8)

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(b * 8, b * 16, kernel_size=3, stride=2, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
            torch.nn.Conv2d(b * 16, b * 16, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.1, inplace=True),
        )

        self.up4 = UpBlock(b * 16, b * 8, b * 8)
        self.up3 = UpBlock(b * 8, b * 4, b * 4)
        self.up2 = UpBlock(b * 4, b * 2, b * 2)
        self.up1 = UpBlock(b * 2, b, b)

        self.final = torch.nn.Conv2d(b, out_channels, kernel_size=1)

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


# ------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------
@st.cache_resource
def load_model_from_clearml() -> UNetConvAE:
    """Скачивает модель из ClearML (если ещё не кэширована) и загружает в память."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Получаем задачу и скачиваем артефакт
    task = Task.get_task(task_id=MODEL_TASK_ID)

    # Ищем артефакт с именем best_model_stage1.pth
    artifact = None
    for art_name, art_obj in task.artifacts.items():
        if art_name == MODEL_ARTIFACT_NAME:
            artifact = art_obj
            break

    if artifact is None:
        st.error(f"Артефакт '{MODEL_ARTIFACT_NAME}' не найден в задаче {MODEL_TASK_ID}")
        st.stop()

    local_path = artifact.get_local_copy()
    model = UNetConvAE(in_channels=3, out_channels=3, base_channels=16)
    state_dict = torch.load(local_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    return model, device


def apply_residual(lq: torch.Tensor, residual_raw: torch.Tensor, residual_scale: float) -> torch.Tensor:
    pred = lq + residual_scale * torch.tanh(residual_raw)
    pred = torch.clamp(pred, 0.0, 1.0)
    return pred


def tiled_inference_residual(
    model: UNetConvAE,
    image_tensor: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    """Tiled inference как в two_stage_one_file.py."""
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
                patch = image_tensor[:, y : y + tile_size, x : x + tile_size]

                if patch.shape[-2:] != (tile_size, tile_size):
                    pad_h = tile_size - patch.shape[-2]
                    pad_w = tile_size - patch.shape[-1]
                    patch = F.pad(patch, (0, pad_w, 0, pad_h), mode="reflect")

                residual = model(patch.unsqueeze(0).to(device)).cpu().squeeze(0)
                residual = residual[:, : min(tile_size, H - y), : min(tile_size, W - x)]

                output[:, y : y + residual.shape[1], x : x + residual.shape[2]] += residual
                weight[:, y : y + residual.shape[1], x : x + residual.shape[2]] += 1.0

    return output / torch.clamp(weight, min=1e-8)


def preprocess_image(pil_image: Image.Image) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize до TARGET_SIZE x TARGET_SIZE, конвертация в тензор [0,1] (C, H, W).
    Возвращает тензор и оригинальный размер (W, H).
    """
    original_size = pil_image.size  # (W, H)

    # Resize до 1000x1000
    img_resized = pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    # Конвертация в numpy array [0,1]
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # HWC -> CHW
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

    return img_tensor, original_size


def restore_aspect_ratio(pil_image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """Возвращает изображение к оригинальным пропорциям, но не сжимает до исходного размера.
    Максимальная сторона остаётся TARGET_SIZE."""
    orig_w, orig_h = original_size
    curr_w, curr_h = pil_image.size  # оба = TARGET_SIZE

    orig_ratio = orig_w / orig_h

    if orig_w >= orig_h:
        new_w = TARGET_SIZE
        new_h = int(TARGET_SIZE / orig_ratio)
    else:
        new_h = TARGET_SIZE
        new_w = int(TARGET_SIZE * orig_ratio)

    if (new_w, new_h) == (curr_w, curr_h):
        return pil_image
    return pil_image.resize((new_w, new_h), Image.LANCZOS)


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Image Restoration",
        page_icon="🔧",
        layout="wide",
    )

    st.title("🔧 Восстановление изображений")
    st.caption("Модель: UNetConvAE (two_stage_one_file.py) через ClearML")

    # Загрузка модели
    with st.spinner("Загрузка модели из ClearML..."):
        model, device = load_model_from_clearml()
    st.success(f"Модель загружена на устройстве: **{device}**")

    # Загрузка изображения
    uploaded_file = st.file_uploader(
        "Загрузите изображение",
        type=["png", "jpeg", "jpg", "bmp", "tif", "tiff", "webp"],
    )

    if uploaded_file is not None:
        # Чтение изображения
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.info(f"Оригинальный размер: {pil_image.size[0]} x {pil_image.size[1]}")

        # Показываем оригинал
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал (LQ)")
            st.image(pil_image, width='stretch')

        # Препроцессинг
        img_tensor, original_size = preprocess_image(pil_image)

        # Inference
        with st.spinner("Восстановление изображения..."):
            residual_raw = tiled_inference_residual(
                model=model,
                image_tensor=img_tensor,
                tile_size=TILE_SIZE,
                overlap=TILE_OVERLAP,
                device=device,
            ).to(device)

            lq_tensor = img_tensor.unsqueeze(0).to(device)
            pred_tensor = apply_residual(
                lq=lq_tensor,
                residual_raw=residual_raw.unsqueeze(0).to(device),
                residual_scale=RESIDUAL_SCALE,
            ).squeeze(0)

            # Конвертация обратно в PIL
            pred_np = pred_tensor.cpu().permute(1, 2, 0).numpy()
            pred_np = np.clip(pred_np * 255.0, 0, 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_np)

            # Возвращаем оригинальные пропорции
            pred_pil = restore_aspect_ratio(pred_pil, original_size)

        # Показываем результат
        with col2:
            st.subheader("Восстановленное (HQ)")
            st.image(pred_pil, width='stretch')

        # Кнопка скачивания
        buf = io.BytesIO()
        pred_pil.save(buf, format="PNG")
        buf.seek(0)

        st.download_button(
            label="📥 Скачать восстановленное изображение",
            data=buf,
            file_name="restored_image.png",
            mime="image/png",
        )


if __name__ == "__main__":
    main()
