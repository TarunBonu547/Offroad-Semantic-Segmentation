# ================================
# INSTALL + IMPORTS
# ================================
!pip install opencv-python

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

print("GPU:", torch.cuda.is_available())

# ================================
# MOUNT GOOGLE DRIVE
# ================================
from google.colab import drive
drive.mount('/content/drive')

# ================================
# EXTRACT DATASET
# ================================
import zipfile

zip_path = "/content/drive/MyDrive/Offroad_Segmentation_Training_Dataset.zip"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/content/dataset")

# ================================
# PATHS
# ================================
train_img_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/train/Color_Images"
train_mask_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/train/Segmentation"

val_img_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/val/Color_Images"
val_mask_dir = "/content/dataset/Offroad_Segmentation_Training_Dataset/val/Segmentation"

# ================================
# DATASET CLASS
# ================================
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = os.listdir(img_dir)

        self.label_map = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            27: 4,
            39: 5
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        # 🔥 BASELINE SIZE
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)

        new_mask = np.zeros_like(mask)
        for k, v in self.label_map.items():
            new_mask[mask == k] = v

        image = image / 255.0

        image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)
        mask = torch.tensor(new_mask, dtype=torch.long)

        return image, mask

# ================================
# DATALOADER
# ================================
train_dataset = SegmentationDataset(train_img_dir, train_mask_dir)
val_dataset = SegmentationDataset(val_img_dir, val_mask_dir)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# ================================
# U-NET MODEL
# ================================
class UNet(nn.Module):
    def __init__(self, num_classes=6):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU()
            )

        self.enc1 = conv_block(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.final(d1)

# ================================
# MODEL + OPTIMIZER + LOSS
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(num_classes=6).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# ================================
# IOU FUNCTION
# ================================
def compute_iou(preds, masks, num_classes=6):
    preds = torch.argmax(preds, dim=1)

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls)
        mask_cls = (masks == cls)

        intersection = (pred_cls & mask_cls).sum().item()
        union = (pred_cls | mask_cls).sum().item()

        if union == 0:
            continue

        ious.append(intersection / union)

    return np.mean(ious)

# ================================
# TRAINING LOOP
# ================================
epochs = 5

for epoch in range(epochs):

    model.train()
    train_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    iou_score = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            iou_score += compute_iou(outputs, masks)

    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"IoU: {iou_score / len(val_loader):.4f}")
    print("-" * 40)