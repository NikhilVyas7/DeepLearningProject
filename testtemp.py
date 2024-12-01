import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_models_pytorch import Unet
from data.datasets import SharedTransformFloodDataset
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex

# Parameters
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/Unet_15.pt"  # Update with the actual path
h, w = 1024, 768
batch_size = 1
num_classes = 10

# Model
model = Unet(in_channels=3, classes=num_classes, activation=None, encoder_name="resnet34")
model = model.to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Transforms
img_transforms = transforms.Compose([
    transforms.ToTensor()
])
label_transforms = transforms.Compose([
    torch.from_numpy
])

# Test DataLoader
val_image_dir = "/home/hice1/nvyas30/scratch/DeepLearningProject/ShrunkenFloodNet/FloodNet-Supervised_v1.0/val/val-org-img"
val_label_dir = "/home/hice1/nvyas30/scratch/DeepLearningProject/ShrunkenFloodNet/FloodNet-Supervised_v1.0/val/val-label-img"

val_dataset = SharedTransformFloodDataset(val_image_dir, val_label_dir, h, w, transform=img_transforms, target_transform=label_transforms)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=5, pin_memory=True)

# Loss function and metric
loss_fn = torch.nn.CrossEntropyLoss()
jaccard_metric = JaccardIndex(num_classes=num_classes, average="macro", task="multiclass").to(device)

# Testing
total_loss = 0
total_mIoU = 0
num_batches = 0

with torch.no_grad():
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()

        # Forward pass
        preds = model(images)
        loss = loss_fn(preds, labels)
        preds = preds.argmax(dim=1)

        # Calculate mIoU
        mIoU = jaccard_metric(preds, labels)

        total_loss += loss.item()
        total_mIoU += mIoU.item()
        num_batches += 1

        print(f"Batch {num_batches}: Loss = {loss.item()}, mIoU = {mIoU.item()}")

# Average metrics
avg_loss = total_loss / num_batches
avg_mIoU = total_mIoU / num_batches

print(f"Testing Complete: Average Loss = {avg_loss}, Average mIoU = {avg_mIoU}")
