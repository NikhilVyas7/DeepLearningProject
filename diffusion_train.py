from PIL import Image
import numpy as np
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from accelerate import Accelerator
from data.datasets import GenerateDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import csv
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
palette = [[0,0,0],[255,0,0],[180,120,120],[160,150,20],[140,140,140],[61,230,250],[0,82,255],[255,0,245],[255,235,0],[4,250,7]]
print("Starting...")
accelerator = Accelerator()

#LORA
lora_config = LoraConfig(
    r=8,  # Low-rank approximation factor
    lora_alpha=32,  # Scaling factor for LoRA weights
    lora_dropout=0.1,  # Dropout for the LoRA layers
    target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Target specific layers (e.g., attention layers)
    bias="none",  # Whether to apply LoRA to bias terms or not
)

controlnet_model = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg",torch_dtype=torch.float32)
#controlnet_model = get_peft_model(controlnet_model, lora_config)

pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",controlnet=controlnet_model,safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)




#Transforms

img_transforms = transforms.Compose([
        transforms.ToTensor()
])
label_transforms = transforms.Compose([
    torch.from_numpy
])

num_epochs = 5
batch_size = 8
dataset = "GenerateFloodNet"
model_name = "StableDiffusion-v-1.5"
learning_rate = 1e-5

print(f"Training {model_name} on {dataset}")
h,w = 512, 512
num_steps = 20

#Maybe something wrong with my output

#Create DataLoaders
val_image_dir = f"/home/hice1/athalanki3/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-org-img"
val_label_dir = f"/home/hice1/athalanki3/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/val/val-label-img"
print(val_image_dir)
val_dataset = GenerateDataset(val_image_dir,val_label_dir,h,w,palette,transform=img_transforms,target_transform=label_transforms)
val_dataloader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=5,pin_memory=True)
#Fails to load the second one (Cuz its around 70 GB of memory allocated on the cpu, holy god)
train_image_dir = f"/home/hice1/athalanki3/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-org-img"
train_label_dir = f"/home/hice1/athalanki3/scratch/DeepLearningProject/{dataset}/FloodNet-Supervised_v1.0/train/train-label-img"
train_dataset = GenerateDataset(train_image_dir,train_label_dir,h,w,palette,transform=img_transforms,target_transform=label_transforms)
train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=5,pin_memory=True)
print("Finished making Dataloaders...")



#Transfer to device

pipe = pipe.to(accelerator.device)

    
#Make optimizer
optimizer = torch.optim.Adam(controlnet_model.parameters(),lr=learning_rate)

#Make loss_fun

loss_fn = torch.nn.MSELoss()
pipe, optimizer, train_dataloader = accelerator.prepare(pipe, optimizer, train_dataloader)
#Create metrics file
# Define the file name and write the header
train_metrics_path = f'running_metrics/training_metrics_{model_name}.csv'
test_metrics_path = f'running_metrics/test_metrics_{model_name}.csv'

with open(train_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch', 'Loss'])  # Header
with open(test_metrics_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Batch'])  # Header

print("Training...")
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch}...")

    #Train
    pipe.controlnet.train()
    for batch_idx, (seg_maps, images) in enumerate(train_dataloader):
        seg_maps, images = seg_maps.to(accelerator.device), images.to(accelerator.device)
        
        latents = pipe.vae.encode(images).latent_dist.sample()

        noise = torch.randn_like(latents)
        bs = latents.shape[0]
        timesteps = torch.randint(0,num_steps,(bs,),device=accelerator.device)
        timesteps = timesteps.long()
        noisy_latents = pipe.scheduler.add_noise(latents,noise,timesteps)

        #Don't really need text-encoder but leaving it here for now
        empty_prompt = [""]*len(seg_maps)

        inputs = pipe.tokenizer(text=empty_prompt, return_tensors="pt", padding=True, truncation=True).to(accelerator.device)
        print(inputs.input_ids)
        exit()
        encoder_hidden_states = pipe.text_encoder(**inputs, return_dict = False)[0]
        down_res, mid_res = pipe.controlnet(noisy_latents,timesteps,encoder_hidden_states=encoder_hidden_states,controlnet_cond=seg_maps,return_dict=False)

        model_pred = pipe.unet(noisy_latents,timesteps,encoder_hidden_states=encoder_hidden_states,return_dict=False,mid_block_additional_residual=mid_res,down_block_additional_residuals=down_res)[0]
        target = pipe.scheduler.get_velocity(latents, noise, timesteps)
        loss = loss_fn(model_pred,target)
        # predicted_images = pipe([""]*len(seg_maps),seg_maps,num_inference_steps=20,output_type="pt").images
        # loss = loss_fn(predicted_images,images) #Not likely the right way
        loss.requires_grad = True
        optimizer.zero_grad()
        accelerator.backward(loss)

        #Convert this script to predicting noise, look at sample notbook provided. #Seems loss actuall decreases here bruh yes!!
        optimizer.step()
        
        print(f"Train: Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
        #Save Metrics
        with open(train_metrics_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, batch_idx + 1, loss.item()])

    #Save current checkpoint
    torch.save(pipe.controlnet.state_dict(),f"checkpoints/{model_name}_{epoch}.pt") 
    
    #Eval
    pipe.controlnet.eval()
    for batch_idx, (seg_maps, images) in enumerate(val_dataloader):
        seg_maps, images = seg_maps.to(accelerator.device), images.to(accelerator.device)

        predicted_images = pipe([""]*len(seg_maps),seg_maps,num_inference_steps=20,output_type="pt").images
        loss = loss_fn(predicted_images,images)

        print(f"Test: Epoch: {epoch}, Batch: {batch_idx}, Test Loss: {loss.item()}")

        #Save test metrics
        with open(test_metrics_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([epoch + 1, batch_idx + 1,loss.item()])



    


print("Complete")