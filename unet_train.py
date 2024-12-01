import os
import torch
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import JaccardIndex
from segmentation_models_pytorch import Unet
from data.datasets import SharedTransformFloodDataset
from models.unet import UNet

# Configuration
print("Starting...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
batch_size = 1
learning_rate = 0.0001384492940061904
num_epochs = 20
num_classes = 10
model_name = "CustomUNet"
h, w = 1024, 768

# Dataset Paths
dataset = "ShrunkenFloodNet"
print(f"Training {model_name} on {dataset}")
train_image_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-org-img"
train_label_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-label-img"
val_image_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-org-img"
val_label_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-label-img"

# Model Definition
if model_name == "Unet":
    model = Unet(in_channels=3, classes=num_classes, activation=None, encoder_name="resnet34")
else:
    model = UNet()

model = model.to(device)
if device == "cuda" and (torch.cuda.device_count() > 1):
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)
    batch_size *= torch.cuda.device_count()

# Loss, Optimizer, and Metrics
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
jaccard_metric = JaccardIndex(num_classes=num_classes, average="macro", task="multiclass").to(device)

# Transforms
img_transforms = transforms.Compose([transforms.ToTensor()])
label_transforms = transforms.Compose([torch.from_numpy])

# Datasets and DataLoaders
train_dataset = SharedTransformFloodDataset(train_image_dir, train_label_dir, h, w, transform=img_transforms, target_transform=label_transforms)
val_dataset = SharedTransformFloodDataset(val_image_dir, val_label_dir, h, w, transform=img_transforms, target_transform=label_transforms)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=5, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

# Metrics File Initialization
train_metrics_path = f'running_metrics/training_metrics_{model_name}.csv'
test_metrics_path = f'running_metrics/test_metrics_{model_name}.csv'
with open(train_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss', 'mIoU'])
with open(test_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss', 'mIoU'])

# Training Loop
print("Training...")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}...")

    # Training Phase
    model.train()
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device).long()
        preds = model(images)
        loss = loss_fn(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics Calculation
        preds = preds.argmax(dim=1)
        mIoU = jaccard_metric(preds, labels)
        print(f"Train: Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item()}, mIoU: {mIoU.item()}")

        # Save Training Metrics
        with open(train_metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, batch_idx + 1, loss.item(), mIoU.item()])

    # Save Checkpoints
    if (epoch + 1) % 5 == 0:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{model_name}_{epoch + 1}.pt")

    # Validation Phase
    model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            images, labels = images.to(device), labels.to(device).long()
            preds = model(images)
            loss = loss_fn(preds, labels)
            test_mIoU = jaccard_metric(preds.argmax(dim=1), labels)
            print(f"Test: Epoch: {epoch + 1}, Batch: {batch_idx + 1}, Loss: {loss.item()}, mIoU: {test_mIoU.item()}")

            # Save Validation Metrics
            with open(test_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1, loss.item(), test_mIoU.item()])

print("Training Complete.")
