# pyTorch_Dataset.py
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# -----------------------------
# Функция предобработки изображения
# -----------------------------
def preprocess_image(path, img_size=128):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype('float32') / 255.0
    img = torch.tensor(img).permute(2, 0, 1)
    return img

# -----------------------------
# Dataset для LQ → HQ по сортировке с разными расширениями
# -----------------------------
class FloorPlanDataset(Dataset):
    def __init__(self, root_dir, img_size=128):
        self.img_size = img_size

        lq_dir = os.path.join(root_dir, 'LQ')
        hq_dir = os.path.join(root_dir, 'HQ')

        # Поддержка jpg и png
        valid_ext = ('.jpg', '.jpeg', '.png')

        self.lq_paths = sorted([os.path.join(lq_dir, f) for f in os.listdir(lq_dir) if f.lower().endswith(valid_ext)])
        self.hq_paths = sorted([os.path.join(hq_dir, f) for f in os.listdir(hq_dir) if f.lower().endswith(valid_ext)])

        if len(self.lq_paths) != len(self.hq_paths):
            raise RuntimeError(f"LQ и HQ имеют разное количество файлов: {len(self.lq_paths)} vs {len(self.hq_paths)}")

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq_img = preprocess_image(self.lq_paths[idx], self.img_size)
        hq_img = preprocess_image(self.hq_paths[idx], self.img_size)
        return lq_img, hq_img

# -----------------------------
# Путь к датасету
# -----------------------------
dataset_root = "C:/Users/speci/.clearml/cache/storage_manager/datasets/ds_11d7ae6a560d4b50aabff2a1a3af6de1"

# -----------------------------
# Создаём датасет
# -----------------------------
full_dataset = FloorPlanDataset(dataset_root, img_size=128)

# -----------------------------
# Разбиваем на train/test
# -----------------------------
train_size = int(0.8 * len(full_dataset))
test_size  = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# -----------------------------
# DataLoaders
# -----------------------------
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -----------------------------
# Проверка
# -----------------------------
lq, hq = next(iter(train_loader))
print("Пример batch LQ:", lq.shape)
print("Пример batch HQ:", hq.shape)
