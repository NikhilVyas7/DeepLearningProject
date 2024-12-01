import optuna
from torch.utils.data import DataLoader
from torchmetrics.classification import JaccardIndex
import csv
import torch
from models.unet import UNet
from segmentation_models_pytorch import Unet
from data.datasets import SharedTransformFloodDataset
from torchvision import transforms
import matplotlib.pyplot as plt

print("Starting", flush=True)
# File to save Optuna results
results_file = "hyperparam_optimization_results.csv"

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    batch_size = 1
    optimizer_type = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    num_classes = 10
    model_name = "Unet"
    h, w = 1024, 768

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the model
    if model_name == "Unet":
        model = Unet(in_channels=3, classes=num_classes, activation=None, encoder_name="resnet34")
    else:
        model = UNet()
    model = model.to(device)

    # Set up optimizer
    if optimizer_type == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    jaccard_metric = JaccardIndex(num_classes=num_classes, average="macro", task="multiclass").to(device)

    # Transforms
    img_transforms = transforms.Compose([transforms.ToTensor()])
    label_transforms = transforms.Compose([torch.from_numpy])

    # Load dataset
    dataset = "ShrunkenFloodNet"
    train_image_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-org-img"
    train_label_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-label-img"
    val_image_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-org-img"
    val_label_dir = f"/home/hice1/nvyas30/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-label-img"
    train_dataset = SharedTransformFloodDataset(train_image_dir, train_label_dir, h, w, transform=img_transforms, target_transform=label_transforms)
    val_dataset = SharedTransformFloodDataset(val_image_dir, val_label_dir, h, w, transform=img_transforms, target_transform=label_transforms)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Training loop (abbreviated)
    num_epochs = 5
    best_mIoU = 0.0  # To track the best mIoU
    results = []  # Store intermediate results

    for epoch in range(num_epochs):
        print("Training")
        model.train()
        total_loss = 0  # Initialize variable to accumulate total loss for the epoch
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device).long()
            preds = model(images)
            loss = loss_fn(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()  # Accumulate the loss

            # Log the training loss and mIoU
            preds = preds.argmax(dim=1)
            mIoU = jaccard_metric(preds, labels)
            # print(f"Train: Trial: {trial.number}, Epoch: {epoch}, Loss: {loss.item()}, mIoU: {mIoU.item()}")

        avg_train_loss = total_loss / len(train_dataloader)  # Calculate the average loss for the epoch

        print("Validating")
        # Validation loop
        model.eval()
        total_mIoU = 0
        with torch.inference_mode():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device).long()
                preds = model(images).argmax(dim=1)
                total_mIoU += jaccard_metric(preds, labels).item()

        avg_mIoU = total_mIoU / len(val_dataloader)

        # Save intermediate results
        print(f"Trial: {trial.number}, Epoch: {epoch}, Learning Rate: {learning_rate:.5f}, "
            f"Batch Size: {batch_size}, Optimizer: {optimizer_type}, Loss: {avg_train_loss:.4f}, Avg mIoU: {avg_mIoU:.4f}")
        results.append({"trial": trial.number, "epoch": epoch, "learning_rate": learning_rate, 
                        "batch_size": batch_size, "optimizer": optimizer_type, "loss": avg_train_loss, "mIoU": avg_mIoU})

        # Report intermediate results to Optuna
        trial.report(avg_mIoU, epoch)

        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Update best mIoU
        if avg_mIoU > best_mIoU:
            best_mIoU = avg_mIoU

    # Save trial-specific results
    trial.set_user_attr("results", results)
    return best_mIoU

# Hyperparameter optimization with Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

# Save results to a file
print("Saving results...")
with open(results_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Add 'loss' in the header to reflect the new saved value
    writer.writerow(["trial", "epoch", "learning_rate", "batch_size", "optimizer", "loss", "mIoU"])  
    for trial in study.trials:
        for epoch_data in trial.user_attrs.get("results", []):  # Load stored results per epoch
            # Add 'loss' to the row that gets written to the CSV file
            writer.writerow([epoch_data["trial"], epoch_data["epoch"], epoch_data["learning_rate"],
                             epoch_data["batch_size"], epoch_data["optimizer"], epoch_data["loss"], epoch_data["mIoU"]])

# Save the best hyperparameters
print("Best hyperparameters:", study.best_params)

# Generate and save Optuna visualizations
print("Generating and saving visualizations...")

# Optimization history plot
history_fig = optuna.visualization.plot_optimization_history(study)
history_fig.write_image("optimization_history.png")

# Parameter importance plot
importance_fig = optuna.visualization.plot_param_importances(study)
importance_fig.write_image("param_importance.png")

print("Visualizations saved as 'optimization_history.png' and 'param_importance.png'.")
