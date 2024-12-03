#Define dataset class
import multiprocessing as mp
import ctypes
import cv2
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import os
from PIL import Image
class FloodDataset(Dataset):
    #This class is the generic FloodDataset with seperate transforms for image and label
    #Cacheing before transforms for now
    def __init__(self,image_dir,label_dir,h,w,transform=None,target_transform=None):
        self.h, self. w =h,w
        self.image_dir = image_dir
        self.label_dir = label_dir

        #Validate all labels exist for every image
        for filepath in os.listdir(image_dir):
            #Validate label exists
            img_num = filepath.split(".")[0]# 7926.png -> 7926
            label = img_num + "_lab.png" #All labels are pngs, while images are jpgs.
            label_path = os.path.join(self.label_dir,label)
            if not os.path.exists(label_path):
                raise Exception(f"Label {label} doesn't exist for {img_num}")
            
        self.images = os.listdir(image_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.use_cache = False
        #Set up cache for images
        self.shared_array_base_images = mp.Array(ctypes.c_uint8, len(self.images) * 3 * h * w)
        shared_array_images = np.ctypeslib.as_array(self.shared_array_base_images.get_obj())
        self.shared_array_images =shared_array_images.reshape(len(self.images),h,w,3)

        #Set up cache for labels
        self.shared_array_base_labels = mp.Array(ctypes.c_uint8, len(self.images) * h * w)
        shared_array_labels = np.ctypeslib.as_array(self.shared_array_base_labels.get_obj())
        self.shared_array_labels = shared_array_labels.reshape(len(self.images),h,w)

        #Allocating the cache for all the individuals is likely really expensive

    def __len__(self):
        return len(self.images)
        
    def set_use_cache(self,use_cache):
        self.use_cache = use_cache
    def __getitem__(self,idx):
        if not self.use_cache:
            image = self.images[idx]
            img_num = image.split(".")[0]# 7926.png -> 7926
            label = img_num + "_lab.png" #All labels are pngs, while images are jpgs.
            label_path = os.path.join(self.label_dir,label)
            image_path = os.path.join(self.image_dir,image)

            #Load images via cv2, and resize them (Some of them are slightly larger)
            loaded_image = cv2.imread(image_path)
            #Switch color channels
            loaded_image = cv2.cvtColor(loaded_image,cv2.COLOR_BGR2RGB)
            loaded_image = cv2.resize(loaded_image, (self.w,self.h))
            loaded_label = cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
            
            loaded_label = cv2.resize(loaded_label, (self.w,self.h), interpolation=cv2.INTER_NEAREST) #Because labels must be specific values

            #Store label and image into cache
            self.shared_array_images[idx] = loaded_image
            self.shared_array_labels[idx] = loaded_label
        else:
            loaded_label = self.shared_array_labels[idx]
            loaded_image = self.shared_array_images[idx]

        #Resize label to have a color channel
        #loaded_image = np.transpose(loaded_image,(0,2,1))
        #Define custom collate function for this i guess
        if self.transform:
            loaded_image = self.transform(loaded_image)
        if self.target_transform:
            loaded_label = self.target_transform(loaded_label)
            
        return loaded_image, loaded_label


class GenerateDataset(FloodDataset):
    def __init__(self,image_dir,label_dir,h,w,palette,transform=None,target_transform=None):
        super(GenerateDataset,self).__init__(image_dir,label_dir,h,w,transform=transform,target_transform=target_transform)
        self.palette = palette
        self.transform = transform
    def  __getitem__(self,idx):
        image, label = super(GenerateDataset,self).__getitem__(idx)
        seg_map = self.convert_label_to_color(label)
        return seg_map, image

    def convert_label_to_color(self,label_img):
        color_seg = np.zeros((self.h,self.w,3))
        for i, color in enumerate(self.palette):
            color_seg[label_img == i,:] = color #label_img here only has 2 channels
        color_seg = np.uint8(color_seg)
        color_seg = self.transform(color_seg)
        return color_seg


class HuggingFaceDataset(FloodDataset):
    def __init__(self,image_dir,label_dir,h,w,palette,transform=None,target_transform=None):
        super(HuggingFaceDataset,self).__init__(image_dir,label_dir,h,w,transform=transform,target_transform=target_transform)
        self.palette = palette
        self.transform = transform
    def  __getitem__(self,idx):
        image, label = super(HuggingFaceDataset,self).__getitem__(idx)
        seg_map = self.convert_label_to_color(label)
        return_dict = {}
        return_dict["pixel_values"] = image
        return_dict["conditioning_pixel_values"] = seg_map
        return_dict["input_ids"] = torch.tensor([49406, 49407])
        return return_dict

    def convert_label_to_color(self,label_img):
        color_seg = np.zeros((self.h,self.w,3))
        for i, color in enumerate(self.palette):
            color_seg[label_img == i,:] = color #label_img here only has 2 channels
        color_seg = np.uint8(color_seg)
        color_seg = self.transform(color_seg)
        return color_seg

    
class SharedTransformFloodDataset(FloodDataset):
    #This class has another parameter, shared_transforms, which are applied between the image and the label, and applied
    #after other transforms
    def __init__(self,image_dir,label_dir,h,w,transform=None,target_transform=None,shared_transform=None):
        super(SharedTransformFloodDataset,self).__init__(image_dir,label_dir,h,w,transform=transform,target_transform=target_transform)
        self.shared_transform = shared_transform
    def __getitem__(self,idx):
        image, label = super(SharedTransformFloodDataset,self).__getitem__(idx)
        if self.shared_transform:
            seed = np.random.randint(1,1000)
            torch.manual_seed(seed)
            image = self.shared_transform(image)
            torch.manual_seed(seed)
            label = self.shared_transform(label)
        return image, label

        
if __name__ == "__main__":
    print("Starting...")



    img_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    label_transforms = transforms.Compose([
        torch.from_numpy
    ])
    image_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/test/test-org-img"
    label_dir = "/home/hice1/athalanki3/scratch/DeepLearningProject/FloodNet/FloodNet-Supervised_v1.0/test/test-label-img"
    test_dataset = SharedTransformFloodDataset(image_dir,label_dir,transform=img_transforms,target_transform=label_transforms)
    

    test_dataloader = DataLoader(test_dataset,batch_size=10,shuffle=False,num_workers=5,pin_memory=True)#Pin_memory makes transfering to
    start_time = time.time()
    imgs, labels = next(iter(test_dataloader))
    end_time = time.time()
    #Img sizes are all 3000,4000.
    print(imgs.shape)
    print(labels.shape)
    print(f"Time: {end_time - start_time}")
    exit()
    print("Using cache....")
    test_dataloader.dataset.set_use_cache(True)

    start_time = time.time()
    imgs, labels = next(iter(test_dataloader))
    end_time = time.time()
    #Img sizes are all 3000,4000.
    print(imgs.shape)
    print(labels.shape)
    print(f"Time: {end_time - start_time}")

    #Might want to consider cacheing after the transform idk, can deal with that later
