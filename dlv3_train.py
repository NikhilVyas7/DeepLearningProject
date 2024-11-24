import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from PIL import Image
from tqdm import tqdm
from torchmetrics.classification import JaccardIndex
from data.datasets import SharedTransformFloodDataset

# Configuration
input_size = (1024, 768)
h, w = input_size
batch_size = 1
epochs = 50
learning_rate = 0.0001
model_name = "DeepLabV3"
num_classes = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

#Create metrics file
# Define the file name and write the header
train_metrics_path = f'running_metrics/training_metrics_{model_name}.csv'
test_metrics_path = f'running_metrics/test_metrics_{model_name}.csv'

with open(train_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss', 'mIoU'])  # Header
with open(test_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'mIoU'])  # Header


img_transforms = transforms.Compose([
        transforms.ToTensor()
])
label_transforms = transforms.Compose([
    torch.from_numpy
])

# Paths to FloodNet dataset
train_image_dir = "ShrunkenFloodNet/FloodNet-Supervised_v1.0/train/train-org-img"
train_mask_dir = "ShrunkenFloodNet/FloodNet-Supervised_v1.0/train/train-label-img"
val_image_dir = "ShrunkenFloodNet/FloodNet-Supervised_v1.0/val/val-org-img"
val_mask_dir = "ShrunkenFloodNet/FloodNet-Supervised_v1.0/val/val-label-img"

# Datasets and DataLoaders
train_dataset = SharedTransformFloodDataset(train_image_dir,train_mask_dir,h,w,transform=img_transforms,target_transform=label_transforms)
val_dataset = SharedTransformFloodDataset(val_image_dir,val_mask_dir,h,w,transform=img_transforms,target_transform=label_transforms)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(train_loader)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn_2 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.conv_3x3_3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn_4 = nn.BatchNorm2d(out_channels)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv_1x1_2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn_5 = nn.BatchNorm2d(out_channels)

        self.conv_1x1_3 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.bn_6 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = F.relu(self.bn_1(self.conv_1x1_1(x)))
        x2 = F.relu(self.bn_2(self.conv_3x3_1(x)))
        x3 = F.relu(self.bn_3(self.conv_3x3_2(x)))
        x4 = F.relu(self.bn_4(self.conv_3x3_3(x)))

        x5 = self.global_avg_pool(x)
        x5 = self.conv_1x1_2(x5)
        x5 = F.relu(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = F.relu(self.bn_6(self.conv_1x1_3(x)))
        return x

# Define the Decoder part for DeepLabV3+
class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_in_channels, low_level_out_channels, out_channels):
        super(DeepLabV3PlusDecoder, self).__init__()
        self.low_level_conv = nn.Conv2d(low_level_in_channels, low_level_out_channels, kernel_size=1, bias=False)
        self.low_level_bn = nn.BatchNorm2d(low_level_out_channels)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(low_level_out_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x, low_level_feat):
        low_level_feat = self.low_level_bn(self.low_level_conv(low_level_feat))
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, low_level_feat], dim=1)
        return self.final_conv(x)

# Combine ASPP and Decoder into DeepLabV3+
class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=num_classes, backbone='resnet50'):
        super(DeepLabV3Plus, self).__init__()

       # Load ResNet50 backbone
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Extract layers for feature extraction
        self.low_level_features = nn.Sequential(*list(self.backbone.children())[:4])  # First few layers (conv1, bn1, relu, maxpool)
        self.high_level_features = nn.Sequential(*list(self.backbone.children())[4:-2])
        
        # ASPP
        self.aspp = ASPP(in_channels=2048, out_channels=256)

        # Decoder
        self.decoder = DeepLabV3PlusDecoder(low_level_in_channels=64, low_level_out_channels=48, out_channels=256)
        
        # Final classification layer
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Extract low-level features
        low_level_feat = self.low_level_features(x)
        
        # Extract high-level features
        x = self.high_level_features(low_level_feat)
        
        # Apply ASPP on high-level features
        x = self.aspp(x)
        
        # Decode features using low-level and ASPP output
        x = self.decoder(x, low_level_feat)
        
        # Final classification layer
        x = self.classifier(x)
        
        # Upsample to match the input image size
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

# Set up the model and print the summary
model = DeepLabV3Plus(num_classes=num_classes)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
jaccard_metric = JaccardIndex(num_classes=num_classes, task="multiclass").to(device)

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    
    # Training Loop
    model.train()
    train_loss = 0.0
    for batch_idx, (images, masks) in enumerate(train_loader):
        images, masks = images.to(device), masks.to(device)
        masks = masks.long()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)

        # Calculate mIoU
        preds = torch.argmax(outputs, dim=1)
        mIoU = jaccard_metric(preds, masks.int())

        #Save Metrics
        with open(train_metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, batch_idx + 1, loss.item(), mIoU.item()])

    # Validation Loop with IoU calculation
    model.eval()
    val_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            masks = masks.long()

            # Generate predictions
            outputs = model(images)

            # Calculate Loss
            loss = criterion(outputs, masks)
            val_loss += loss.item() * images.size(0)

            # Calculate mIoU
            preds = torch.argmax(outputs, dim=1)
            mIoU = jaccard_metric(preds, masks.int())
            total_iou += mIoU

            # Save IoU metrics to CSV
            with open(test_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1, mIoU.item()])

    jaccard_metric.reset()

    # Average validation loss and IoU for the epoch
    val_loss /= len(val_loader.dataset)
    mean_iou = total_iou / len(val_loader)
    print(f"Validation Loss: {val_loss:.4f}, Mean IoU: {mean_iou:.4f}")

print("Training complete.")
