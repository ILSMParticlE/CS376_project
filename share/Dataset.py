import numpy as np
import scipy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torch.utils.data.sampler import SubsetRandomSampler
from glob import glob
import csv
import random
import matplotlib.pyplot as plt
from PIL import Image

# transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
transform = T.Compose([T.Resize(256), T.RandomCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
test_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

center_transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), T.Grayscale(num_output_channels=1)])
gray_transform = T.Compose([T.Resize(400), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), T.Grayscale(num_output_channels=1)])

def head_tensor(img):
    img_tensor = gray_transform(img)
    h_start = 20
    w_start = 80
    img_tensor = img_tensor[:, h_start:224+h_start, w_start:224+w_start]
    return img_tensor

def hand_tensor(img):
    img_tensor = gray_transform(img)
    img_tensor = img_tensor[:, -224:, -224:]
    return img_tensor

def jacobkie_transform(img):
    return torch.stack([center_transform(img).squeeze(), head_tensor(img).squeeze(), hand_tensor(img).squeeze()], dim=0)


def transform_data(crop_mode, affine, color):
    crop_n = 3
    condition = [0] * (crop_n + 2)
    condition[crop_mode-1] = 1
    condition[crop_n] = affine
    condition[crop_n+1] = color

    
    # probability of affine and light transform
    p_affine, p_light = 0.75, 0.75

    # maximal scale of crop
    min_randcrop_scale = 0.7
    
    # maximal change of light
    max_light = 0.2
    
    # maximal change of affine
    max_rotation = 10
    max_scale = 1.1
    max_shear = 15
    
    transform =  [
        # crops
        T.CenterCrop(224),
        T.RandomCrop(224),
        T.RandomResizedCrop(224, scale = (min_randcrop_scale, 1)),
        
        # random affine
        T.RandomApply([T.RandomAffine(max_rotation, translate = None, scale = (1, max_scale), shear = max_shear)], p_affine),
        #T.RandomAffine(max_rotation, translate = None, scale = (1, max_scale), shear = max_shear),
        # random light
        T.RandomApply([T.ColorJitter(max_light, max_light, max_light, max_light)], p_light)
        #T.ColorJitter(max_light, max_light, max_light, max_light)
        
    ]
    for i in range(len(condition)):
        if not condition[i]: transform[i] = False
    transform = list(filter(lambda x : x, transform))
    print(transform)
    transform = [T.Resize(256)] + transform + [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    return T.Compose(transform)



class trainDataset(Dataset):
    '''
    dir_data : directory where input data locates (i.e. "train" or "test" or some sample images)
    mode_test : True for test, False for train (default = False)
    '''
    def __init__(self, dir_data, mode_test=False, crop_mode=1, affine=False,  color=False):
        self.mode_test = mode_test
        
        # datasets
        imageList = []

        data = glob('./' + dir_data)
        print(data)
        for dir in data:
            for img in glob(dir + '/*jpg'):
                imageList.append(img)
        self.image_list = imageList
        
        # labels 
        if not mode_test:
            with open('./driver_imgs_list.csv') as f:
                labels = list(csv.reader(f))[1:]
                self.labels = {label[2] : label[1] for label in labels}
                # print(self.labels)
                f.close()
        
        
        # preprocessor
        self.preprocessing = transform_data(crop_mode, affine, color)
        self.test_preprocessing = test_transform
        
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img = Image.open(self.image_list[idx]).convert('RGB')
        if not self.mode_test:
            img_tensor = self.preprocessing(img)
            label_idx = self.image_list[idx].split('\\')[-1]
            label = int(self.labels[label_idx][1:])
            return img_tensor, label
        img_tensor = self.test_preprocessing(img)
        return img_tensor, self.image_list[idx].split('/')[-1].split('\\')[-1]




def split_dataset(dataset, train_size=0.8):

    data_len = len(dataset)
    indexes = list(range(data_len))
    random.shuffle(indexes)
    train_anchor = int(data_len * train_size)

    train_sampler = SubsetRandomSampler(indexes[:train_anchor])
    valid_sampler = SubsetRandomSampler(indexes[train_anchor:])

    return train_sampler, valid_sampler