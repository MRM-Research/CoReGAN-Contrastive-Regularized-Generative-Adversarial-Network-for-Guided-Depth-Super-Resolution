from .misc import tan_fi
import numpy as np
import os
import torch
import torch.nn.functional as F



import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2

class Dataset():  
    
    def __init__(
            self,
            rgb_dir: str,
            target_dir: str,
            augmentation=None, 
            preprocessing=None,
            resize=None,
    ):
        self.rgb_list = self.load_img(rgb_dir)
        self.target_list = self.load_img(target_dir)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.resize = resize
        
    def extract_number(self, filename):
        return int(filename.split('_')[1].split('.')[0])
    
    def standardize(self,image,mean,std):
        image = image/255.0
        image_normalised = image - mean
        image_standardized = image_normalised / std
        
        return image_standardized

    def standardize_depth(self, image, global_min_depth, global_max_depth):
        # Scale and shift to [-1, 1]
        image_normalized = 2 *((image - global_min_depth) / (global_max_depth - global_min_depth)) - 1
        return image_normalized
    
    def standardize_low_res(self, image, global_min_depth, global_max_depth):
        # Scale and shift to [0, 1]
        image_normalized = ((image - global_min_depth) / (global_max_depth - global_min_depth)) 
        return image_normalized

    def load_img(self, directory):
        img_list = []
        files = os.listdir(directory)
        for file in files:
            if file.endswith(".npy"):
                img_list.append(os.path.join(directory, file))
                
        sorted_img_list = sorted(img_list, key=self.extract_number)
        return sorted_img_list
    
    def __len__(self):
        return len(self.rgb_list)

    def __getitem__(self, i):
        # read data
        rgb_image = np.load(self.rgb_list[i])
        target_image = np.load(self.target_list[i])
        rgb_image = rgb_image.astype(np.float32)
        
        depth_low_res_image = None  # Initialize to None
        
        if self.augmentation:
            augmented = self.augmentation(image=rgb_image, target=target_image)
            rgb_image, target_image = augmented['image'], augmented['target']
        
        if self.resize:
            transform = self.resize(image=target_image)
            depth_low_res_image = transform['image']
            depth_low_res_image = np.array(depth_low_res_image)

        if self.preprocessing:
            rgb_image = np.transpose(rgb_image, (2, 0, 1))
            rgb_image[0] = self.standardize(rgb_image[0],0.48057137,0.28918139)
            rgb_image[1] = self.standardize(rgb_image[1],0.4109165,0.29590342)
            rgb_image[2] = self.standardize(rgb_image[2],0.39225202,0.30930299)
            
        target_image = self.standardize_depth(target_image.astype(np.float32),0.7132995,9.99547)
        depth_low_res_image = self.standardize_low_res(depth_low_res_image.astype(np.float32),0.7132995,9.99547)
        
        target_image = target_image[np.newaxis, :]
        
        if depth_low_res_image is not None:
            depth_low_res_image = depth_low_res_image[np.newaxis, :]
            depth_low_res_image = torch.from_numpy(depth_low_res_image)
        rgb_image = torch.from_numpy(rgb_image)
        target_image = torch.from_numpy(target_image)
        
        return rgb_image, depth_low_res_image, target_image


