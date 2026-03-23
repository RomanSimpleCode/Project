import os
import cv2
import random
import numpy as np
from tqdm import tqdm

# -----------------------------
# Пути к исходному и новому датасету
# -----------------------------

# Мой скаченный датасет лежит по этому пути, если путь другой, то нужно поменять
input_root  = r"C:\Users\speci\.clearml\cache\storage_manager\datasets\ds_11d7ae6a560d4b50aabff2a1a3af6de1"
# Датасет который создаётся после предобработки
output_root = r"D:\Projects\floorplan_dataset"

lq_dir = os.path.join(input_root, "LQ")
hq_dir = os.path.join(input_root, "HQ")

# -----------------------------
# Параметры
# -----------------------------
target_size = 2000
split_ratio = 0.8
random.seed(42)

valid_ext = (".jpg", ".jpeg", ".png")

# -----------------------------
# Функция padding
# -----------------------------
def pad_to_size(img, size):
    h, w = img.shape[:2]
    pad_h = max(size - h, 0)
    pad_w = max(size - w, 0)
    padded = np.zeros((size, size, 3), dtype=img.dtype)
    padded[:h, :w] = img
    return padded

# -----------------------------
# Сопоставляем пары
# -----------------------------
lq_files = [f for f in os.listdir(lq_dir) if f.lower().endswith(valid_ext)]
hq_files = [f for f in os.listdir(hq_dir) if f.lower().endswith(valid_ext)]

hq_dict = {os.path.splitext(f)[0]: f for f in hq_files}

pairs = []
for lq_file in lq_files:
    name = os.path.splitext(lq_file)[0]
    if name in hq_dict:
        pairs.append((lq_file, hq_dict[name]))

print(f"Найдено совпадающих пар: {len(pairs)}")

# -----------------------------
# Разделение train/test
# -----------------------------
random.shuffle(pairs)
split_idx = int(len(pairs) * split_ratio)
train_pairs = pairs[:split_idx]
test_pairs  = pairs[split_idx:]

# -----------------------------
# Создаём папки
# -----------------------------
for split in ["train", "test"]:
    for t in ["LQ", "HQ"]:
        os.makedirs(os.path.join(output_root, split, t), exist_ok=True)

# -----------------------------
# Функция обработки и сохранения
# -----------------------------
def process_and_save(pairs, split):
    for lq_file, hq_file in tqdm(pairs, desc=f"{split}"):
        lq_path = os.path.join(lq_dir, lq_file)
        hq_path = os.path.join(hq_dir, hq_file)

        lq_img = cv2.imread(lq_path)
        hq_img = cv2.imread(hq_path)
        if lq_img is None or hq_img is None:
            continue

        # BGR → RGB
        lq_img = cv2.cvtColor(lq_img, cv2.COLOR_BGR2RGB)
        hq_img = cv2.cvtColor(hq_img, cv2.COLOR_BGR2RGB)

        # padding до target_size
        lq_img = pad_to_size(lq_img, target_size)
        hq_img = pad_to_size(hq_img, target_size)

        name = os.path.splitext(lq_file)[0]

        # сохраняем: LQ → JPG, HQ → PNG
        cv2.imwrite(os.path.join(output_root, split, "LQ", name + ".jpg"), cv2.cvtColor(lq_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_root, split, "HQ", name + ".png"), cv2.cvtColor(hq_img, cv2.COLOR_RGB2BGR))

# -----------------------------
# Обработка train/test
# -----------------------------
process_and_save(train_pairs, "train")
process_and_save(test_pairs, "test")

print("Готово! Датасет подготовлен")
