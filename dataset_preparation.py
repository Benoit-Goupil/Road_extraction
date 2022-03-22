import torch 
import os
import cv2
import random

import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms




CURRENT_PATH = os.path.curdir
DATA_PATH = os.path.join(CURRENT_PATH,"data")

def get_random_crop(image, mask, crop_height, crop_width):

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)

    crop_image = image[y: y + crop_height, x: x + crop_width]
    crop_mask = mask[y: y + crop_height, x: x + crop_width]

    return crop_image, crop_mask

def train_transform(image, mask):

        image, mask = get_random_crop(image, mask, 256, 256)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)

        # Random vertical flipping
        if random.random() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        if random.random() > 0.5:
            image = np.rot90(image)
            mask = np.rot90(mask)
        
        return image, mask

def validation_transform(image, mask):
    image = cv2.copyMakeBorder(image, 18, 18, 18, 18, cv2.BORDER_CONSTANT, (255, 255, 255))
    mask = cv2.copyMakeBorder(mask, 18, 18, 18, 18, cv2.BORDER_CONSTANT, (0, 0, 0))
    return image, mask


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map





class Road_Dataset(Dataset):
    def __init__(self, meta_df, data_path = DATA_PATH, transformation=None, split='train'):
        super().__init__()
        self.transformation = transformation
        self.meta_df = meta_df
        self.data_splited = meta_df[meta_df['split'] == split]
        self.mask_list = self.data_splited['tif_label_path'].to_list()
        self.images_list = self.data_splited['tiff_image_path'].to_list()
        #paths of images and masks
        self.image_mask_path_list = [(os.path.join(data_path, str(image_name)), os.path.join(data_path, str(mask_name))) for image_name, mask_name in zip(self.images_list, self.mask_list)]
    
    def __len__(self):
        return len(self.data_splited)

    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()
        img_sat = cv2.imread(self.image_mask_path_list[i][0])
        mask = cv2.imread(self.image_mask_path_list[i][1])


        if self.transformation:
            img_sat, mask = self.transformation(img_sat, mask)
        
        img_sat = np.array(img_sat)/255.0
        mask = one_hot_encode(mask, [0, 255])
        
        img_sat = transforms.ToTensor()(img_sat.astype(np.float32))
        mask = transforms.ToTensor()(mask.astype(np.float32))

        return img_sat, mask

