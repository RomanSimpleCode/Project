"""
Streamlit приложение для восстановления изображений через ClearML модель.
Модель: CleanResidualUNet (two_stage_one_file.py)
"""

import io
import sys

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from clearml import Task
from PIL import Image
from pytorch_msssim import ssim
from two_stage_one_file import CONFIG, CleanResidualUNet

# ------------------------------------------------------------
# Windows guard для ClearML
# ------------------------------------------------------------
if sys.platform == "win32":
    Task.ignore_requirements("pywin32")

# ------------------------------------------------------------
# Константы
# ------------------------------------------------------------
MODEL_TASK_ID = "bc17547fbb28414c80122a671060c2f4"
MODEL_ARTIFACT_NAME = "best_model"
TARGET_SIZE = CONFIG.get("image_size", 1024)
TILE_SIZE = 512
TILE_OVERLAP = 64
CLASSICAL_METHODS = {
    "lanczos": "Lanczos resampling",
    "bicubic": "Bicubic interpolation",
    "tv": "Total Variation regularization",
    "nedi": "NEDI (edge-directed, approximation)",
}


# ------------------------------------------------------------
# Утилиты
# ------------------------------------------------------------
@st.cache_resource
def load_model_from_clearml() -> tuple:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    task = Task.get_task(task_id=MODEL_TASK_ID)

    artifact = None
    for art_name, art_obj in task.artifacts.items():
        if art_name == MODEL_ARTIFACT_NAME:
            artifact = art_obj
            break

    if artifact is None:
        st.error(f"Артефакт '{MODEL_ARTIFACT_NAME}' не найден в задаче {MODEL_TASK_ID}")
        st.stop()

    local_path = artifact.get_local_copy()
    model = CleanResidualUNet(
        in_ch=3,
        out_ch=3,
        base=CONFIG.get("base_channels", 64),
        dropout=CONFIG.get("dropout", 0.0),
        extra_bottleneck_blocks=CONFIG.get("extra_bottleneck_blocks", 2),
    )

    checkpoint = torch.load(local_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, device


def tiled_inference(
    model: CleanResidualUNet,
    image_tensor: torch.Tensor,
    tile_size: int,
    overlap: int,
    device: torch.device,
) -> torch.Tensor:
    c, H, W = image_tensor.shape
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError("tile_size must be > overlap")

    output = torch.zeros((c, H, W), dtype=torch.float32)
    weight = torch.zeros((c, H, W), dtype=torch.float32)

    ys = list(range(0, max(H - tile_size + 1, 1), stride))
    xs = list(range(0, max(W - tile_size + 1, 1), stride))

    if not ys or ys[-1] != H - tile_size:
        ys.append(max(H - tile_size, 0))
    if not xs or xs[-1] != W - tile_size:
        xs.append(max(W - tile_size, 0))

    with torch.no_grad():
        for y in ys:
            for x in xs:
                patch = image_tensor[:, y : y + tile_size, x : x + tile_size]

                ph, pw = patch.shape[-2], patch.shape[-1]
                if ph != tile_size or pw != tile_size:
                    patch = F.pad(patch, (0, tile_size - pw, 0, tile_size - ph), mode="reflect")

                pred, _ = model(patch.unsqueeze(0).to(device))
                pred = torch.clamp(pred, 0.0, 1.0).cpu().squeeze(0)
                pred = pred[:, :ph, :pw]

                output[:, y : y + ph, x : x + pw] += pred
                weight[:, y : y + ph, x : x + pw] += 1.0

    return output / torch.clamp(weight, min=1e-8)


def preprocess_image(pil_image: Image.Image) -> tuple:
    original_size = pil_image.size
    img_resized = pil_image.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    return img_tensor, original_size


def restore_aspect_ratio(pil_image: Image.Image, original_size: tuple) -> Image.Image:
    orig_w, orig_h = original_size
    orig_ratio = orig_w / orig_h

    if orig_w >= orig_h:
        new_w = TARGET_SIZE
        new_h = int(TARGET_SIZE / orig_ratio)
    else:
        new_h = TARGET_SIZE
        new_w = int(TARGET_SIZE * orig_ratio)

    if (new_w, new_h) == pil_image.size:
        return pil_image
    return pil_image.resize((new_w, new_h), Image.LANCZOS)


def get_output_size(original_size: tuple) -> tuple:
    orig_w, orig_h = original_size
    orig_ratio = orig_w / orig_h
    if orig_w >= orig_h:
        return TARGET_SIZE, int(TARGET_SIZE / orig_ratio)
    return int(TARGET_SIZE * orig_ratio), TARGET_SIZE


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image_np = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image_np)


def pil_to_tensor(pil_image: Image.Image) -> torch.Tensor:
    image_np = np.array(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(image_np).permute(2, 0, 1)


def resize_with_method(image: Image.Image, size: tuple, method: str) -> Image.Image:
    if method == "lanczos":
        return image.resize(size, Image.LANCZOS)
    if method == "bicubic":
        return image.resize(size, Image.BICUBIC)
    raise ValueError(f"Unknown resize method: {method}")


def tv_regularization(image: np.ndarray, weight: float = 0.12) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    denoised = cv2.fastNlMeansDenoisingColored(
        image_u8, None, h=8, hColor=8, templateWindowSize=7, searchWindowSize=21
    )
    blended = cv2.addWeighted(image_u8, 1.0 - weight, denoised, weight, 0)
    return blended.astype(np.float32) / 255.0


def nedi_approximation(image: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    filtered = cv2.edgePreservingFilter(image_u8, flags=1, sigma_s=60, sigma_r=0.4)
    sharpened = cv2.addWeighted(image_u8, 0.35, filtered, 0.65, 0)
    return sharpened.astype(np.float32) / 255.0


def run_classical_method(method: str, input_image: Image.Image, original_size: tuple) -> Image.Image:
    output_size = get_output_size(original_size)

    if method in {"lanczos", "bicubic"}:
        return resize_with_method(input_image, output_size, method)

    base_resized = input_image.resize(output_size, Image.BICUBIC)
    image_np = np.array(base_resized, dtype=np.float32) / 255.0
    if method == "tv":
        restored_np = tv_regularization(image_np)
    elif method == "nedi":
        restored_np = nedi_approximation(image_np)
    else:
        raise ValueError(f"Unsupported classical method: {method}")

    return Image.fromarray(np.clip(restored_np * 255.0, 0, 255).astype(np.uint8))


def prepare_reference_image(reference_image: Image.Image, target_size: tuple) -> Image.Image:
    return reference_image.convert("RGB").resize(target_size, Image.LANCZOS)


def line_mask_tensor(target: torch.Tensor, white_threshold: float = 245 / 255.0) -> torch.Tensor:
    gray = target.mean(dim=1, keepdim=True)
    mask = (gray < white_threshold).float()
    return F.max_pool2d(mask, kernel_size=5, stride=1, padding=2)


def sobel_edges_tensor(image: torch.Tensor) -> torch.Tensor:
    gray = image.mean(dim=1, keepdim=True)
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def calc_line_l1_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    mask = line_mask_tensor(target)
    value = (torch.abs(pred - target) * mask).sum() / (mask.sum() * pred.shape[1] + 1e-6)
    return float(value.item())


def weighted_l1_metric_tensor(pred: torch.Tensor, target: torch.Tensor) -> float:
    mask_weight = float(CONFIG.get("line_mask_weight", 10.0))
    mask = line_mask_tensor(target)
    value = (torch.abs(pred - target) * (1.0 + mask_weight * mask)).mean()
    return float(value.item())


def compute_metrics(
    pred_image: Image.Image,
    target_image: Image.Image,
    baseline_image: Image.Image | None = None,
) -> dict:
    pred_tensor = pil_to_tensor(pred_image).unsqueeze(0)
    target_tensor = pil_to_tensor(target_image).unsqueeze(0)

    mse = torch.mean((pred_tensor - target_tensor) ** 2).item()
    mae = torch.mean(torch.abs(pred_tensor - target_tensor)).item()
    psnr = 100.0 if mse <= 1e-12 else 20.0 * np.log10(1.0 / np.sqrt(mse))
    ssim_value = float(ssim(pred_tensor, target_tensor, data_range=1.0, size_average=True).item())
    line_l1 = calc_line_l1_tensor(pred_tensor, target_tensor)
    weighted_l1 = weighted_l1_metric_tensor(pred_tensor, target_tensor)
    edge_ref = float(F.l1_loss(sobel_edges_tensor(pred_tensor), sobel_edges_tensor(target_tensor)).item())

    gain_line = np.nan
    if baseline_image is not None:
        baseline_tensor = pil_to_tensor(baseline_image.resize(pred_image.size, Image.LANCZOS)).unsqueeze(0)
        baseline_line_l1 = calc_line_l1_tensor(baseline_tensor, target_tensor)
        gain_line = baseline_line_l1 - line_l1

    return {
        "PSNR": psnr,
        "SSIM": ssim_value,
        "MSE": mse,
        "MAE": mae,
        "Line L1": line_l1,
        "Weighted L1": weighted_l1,
        "Gain Line": gain_line,
        "Edge ref": edge_ref,
    }


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
    st.caption("Модель: CleanResidualUNet (two_stage_one_file.py) через ClearML")

    with st.spinner("Загрузка модели из ClearML..."):
        model, device = load_model_from_clearml()
    st.success(f"Модель загружена на устройстве: **{device}**")

    uploaded_file = st.file_uploader(
        "Загрузите изображение",
        type=["png", "jpeg", "jpg", "bmp", "tif", "tiff", "webp"],
    )

    reference_file = st.file_uploader(
        "Опционально: загрузите эталонное HQ-изображение для расчёта метрик",
        type=["png", "jpeg", "jpg", "bmp", "tif", "tiff", "webp"],
    )

    if uploaded_file is not None:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.info(f"Оригинальный размер: {pil_image.size[0]} x {pil_image.size[1]}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Оригинал (LQ)")
            st.image(pil_image, width='stretch')

        selected_methods = st.multiselect(
            "Классические методы для сравнения",
            options=list(CLASSICAL_METHODS.keys()),
            default=["lanczos", "bicubic"],
            format_func=lambda key: CLASSICAL_METHODS[key],
        )
        run_comparison = st.button("Запустить сравнение", use_container_width=True)
        st.caption("NEDI реализован как edge-directed approximation без отдельной нейросети.")

        img_tensor, original_size = preprocess_image(pil_image)

        with st.spinner("Восстановление изображения..."):
            pred_tensor = tiled_inference(
                model=model,
                image_tensor=img_tensor,
                tile_size=TILE_SIZE,
                overlap=TILE_OVERLAP,
                device=device,
            )

            pred_pil = tensor_to_pil(pred_tensor)
            pred_pil = restore_aspect_ratio(pred_pil, original_size)

        with col2:
            st.subheader("Восстановленное (HQ)")
            st.image(pred_pil, width='stretch')

        reference_pil = None
        baseline_pil = None
        if reference_file is not None:
            reference_pil = prepare_reference_image(Image.open(reference_file), pred_pil.size)
            baseline_pil = pil_image.convert("RGB").resize(pred_pil.size, Image.LANCZOS)
            st.subheader("Эталонное HQ для метрик")
            st.image(reference_pil, width="stretch")

        metrics_rows: list[dict] = []
        if reference_pil is not None:
            nn_metrics = compute_metrics(pred_pil, reference_pil, baseline_pil)
            metrics_rows.append({"Method": "CleanResidualUNet", **nn_metrics})

        if run_comparison:
            if not selected_methods:
                st.warning("Выберите хотя бы один классический метод для сравнения.")
            else:
                st.subheader("Сравнение с классическими методами")
                comparison_columns = st.columns(max(1, len(selected_methods)))
                for idx, method in enumerate(selected_methods):
                    result_image = run_classical_method(method, pil_image, original_size)
                    with comparison_columns[idx]:
                        st.markdown(f"**{CLASSICAL_METHODS[method]}**")
                        st.image(result_image, width="stretch")

                    row = {
                        "Method": CLASSICAL_METHODS[method],
                        "PSNR": np.nan,
                        "SSIM": np.nan,
                        "MSE": np.nan,
                        "MAE": np.nan,
                        "Line L1": np.nan,
                        "Weighted L1": np.nan,
                        "Gain Line": np.nan,
                        "Edge ref": np.nan,
                    }
                    if reference_pil is not None:
                        row.update(compute_metrics(result_image, reference_pil, baseline_pil))
                    metrics_rows.append(row)

        if metrics_rows:
            st.subheader("Метрики")
            st.dataframe(metrics_rows, width="stretch")
        else:
            st.info("Чтобы получить PSNR/SSIM/MSE/MAE, загрузите эталонное HQ-изображение.")

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
