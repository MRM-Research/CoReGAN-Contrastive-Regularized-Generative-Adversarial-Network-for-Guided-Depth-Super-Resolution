from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import cv2
from misc import tan_fi
import numpy as np
import torch
import os


class Dataset():  
    
    def __init__(
            self,
            rgb_dir: str,
            target_dir: str,
            augmentation=None, 
            preprocessing=None,
            resize = None,
        
    ):
        self.rgb_list = self.load_img(rgb_dir)
        self.target_list = self.load_img(target_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize = resize
        
    def extract_number(self,filename):
        return int(filename.split('_')[1].split('.')[0])
        
        
    def standardize(self,image,mean,std):
        image = image/255.0
        image_normalised = image - mean
        image_standardized = image_normalised / std
        
        return image_standardized
    
    def __getitem__(self, i):
        
        # read data
        print(self.rgb_list[i])
        print(self.target_list[i])
        rgb_image = np.load(self.rgb_list[i])
        target_image = np.load(self.target_list[i])
        
        if self.augmentation:
            
            augmented = self.augmentation(image=rgb_image, target=target_image)
            rgb_image,target_image = augmented['image'],augmented['target']
            
            if self.resize:
                transform = self.resize(image = target_image)
                depth_low_res_image = transform['image']
                depth_low_res_image = np.array(depth_low_res_image)
        
                
        target_image = np.array(target_image)
        
        if self.preprocessing:
            
            rgb_image[0] = self.standardize(rgb_image[0],0.48057137,0.28918139)
            rgb_image[1] = self.standardize(rgb_image[1],0.4109165,0.29590342)
            rgb_image[2] = self.standardize(rgb_image[2],0.39225202,0.30930299)
            target_image = target_image/255.0
            depth_low_res_image[0] = self.standardize(depth_low_res_image[0],0.42791329,0.5207557)
            depth_low_res_image[1] = self.standardize(depth_low_res_image[1],0.42791329,0.5207557)
            depth_low_res_image[2] = self.standardize(depth_low_res_image[2],0.42791329,0.5207557)

        target_image = torch.from_numpy(target_image)
        depth_low_res_image = torch.from_numpy(depth_low_res_image)
        rgb_image = torch.from_numpy(rgb_image)
        target_image = target_image.unsqueeze(0)
        depth_low_res_image = depth_low_res_image.unsqueeze(0)
        
        target_image = tan_fi(target_image)
        
        #target_image = target_image.repeat(3,1,1)
        #depth_low_res_image = depth_low_res_image.repeat(3,1,1)
        
        return rgb_image, depth_low_res_image, target_image
    
    def load_img(self,directory):
        img_list = []
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".npy"):
                img_list.append(os.path.join(directory, file))
                
        sorted_img_list = sorted(img_list, key=self.extract_number)
        return sorted_img_list
    
    def __len__(self):
        return len(self.rgb_list)
