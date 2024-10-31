from data.datasets import SharedTransformFloodDataset
from models.unet import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.classification import JaccardIndex
import csv
import torch


print("Starting...")


device = "cuda" if torch.cuda.is_available() else "cpu"

batch_size = 1
#GPU's really don't have enough memory, should try using portions of the image and label.
#Takes a lot of memory, especially for the backward pass. Should definetly try shrinking the images

learning_rate = 1e-3 #Switch to use learning rate scheduler eventually
num_epochs = 20
num_classes = 10
model = UNet().to(device)
model_name = "UNetCustom"
#I should later switch the UNet model to use patches

#Use DataParallel if more than one device
if device == "cuda" and (torch.cuda.device_count() > 1):
    model = torch.nn.DataParallel(model)
    batch_size *= torch.cuda.device_count()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
jaccard_metric = JaccardIndex(num_classes=num_classes,average="macro",task="multiclass").to(device)
#Transforms

img_transforms = transforms.Compose([
        transforms.ToTensor()
])
label_transforms = transforms.Compose([
    torch.from_numpy
])


#Create DataLoaders
val_image_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/val/val-org-img"
val_label_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/val/val-label-img"
val_dataset = SharedTransformFloodDataset(val_image_dir,val_label_dir,transform=img_transforms,target_transform=label_transforms)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=5,pin_memory=True)
#Fails to load the second one (Cuz its around 70 GB of memory allocated on the cpu, holy god)
train_image_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/train/train-org-img"
train_label_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/train/train-label-img"
train_dataset = SharedTransformFloodDataset(train_image_dir,train_label_dir,transform=img_transforms,target_transform=label_transforms)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True)
print("Finished making Dataloaders...")

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

print("Training...")

for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}...")

    #Train
    model.train()
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        labels = labels.long()
        #Forward

        preds = model(images)

        #Backward
        loss = loss_fn(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #Calc Metrics
        mIoU = jaccard_metric(preds,labels)
        print(f"Train: Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}, mIoU: {mIoU.item()}")

        #Save Metrics
        with open(train_metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, batch_idx + 1, loss.item(), mIoU.item()])

    #Save model checkpoint after each batch of training
    torch.save(model,f"checkpoints/{model_name}_{epoch}.pt")    

    #Test
    model.eval()
    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(val_dataloader):
            images, labels = images.to(device), labels.to(device)

            preds = model(images)
            test_mIoU = jaccard_metric(preds,labels)
            print(f"Test: Epoch: {epoch}, Batch: {batch_idx}, mIoU: {test_mIoU.item()}")

            #Save Metrics
            with open(test_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1, test_mIoU.item()])


print("Complete")