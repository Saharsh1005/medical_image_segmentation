# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:55:50 2022

@author: Saharsh
"""
import os
import numpy as np
import pandas as pd 
from patches import create_patches
import cv2
from matplotlib import pyplot as plt
import random
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import albumentations as A
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp

from dataset import NuclearSegmentationDataset
from utils import visualize
from unet3plus import UNet3Plus

from train import train
from validate import validate
from loss_functions import FocalLoss, DiceLoss


MAIN_DIR="C:/Users/Saharsh/Desktop/Onward/Work/medical_image_segmentation"

i_x = os.listdir(os.path.join(MAIN_DIR,"tissue_images"))
m_x = os.listdir(os.path.join(MAIN_DIR,"tissue_masks"))
i_x.sort()
m_x.sort()

ALL_IMAGES=[os.path.join(MAIN_DIR,"tissue_images",x) for x in i_x]
ALL_MASKS=[os.path.join(MAIN_DIR,"tissue_masks",x) for x in m_x]

img_patches = []
mask_patches=[]

for i,m in zip(ALL_IMAGES,ALL_MASKS):
    img_patches.extend(create_patches(i, 256,256,read_as_rgb_bw=1))
    mask_patches.extend(create_patches(m, 256,256,read_as_rgb_bw=0))
    

final_images = np.array(img_patches)
final_masks = np.array(mask_patches)
final_masks = np.expand_dims(final_masks, -1)
print(final_images.shape,final_masks.shape)

#Splitting dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(final_images, final_masks, test_size = 0.2, random_state = 0)
#print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

train_transform = A.Compose(
    [          
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
          ], p=0.5),
        A.CLAHE(p=0.5),
        A.RandomBrightnessContrast(p=0.5),    
    ])

train_dataset = NuclearSegmentationDataset(X_train,y_train,transform=train_transform)
test_dataset = NuclearSegmentationDataset(X_test,y_test,transform=None)

train_data_loader = torch.utils.data.DataLoader(train_dataset,batch_size=4, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

test_data_loader = torch.utils.data.DataLoader(test_dataset,batch_size=4, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

# Check Tensor shapes 
batch = next(iter(train_data_loader))
images, labels = batch
print(images.shape, labels.shape, type(images), type(labels), images.dtype, labels.dtype)

# Sanity check sample n=1 
testImg = images[1]
testMsk = labels[1]
print("Img: ",testImg.shape, testImg.dtype, type(testImg))
print("Mask: ",testMsk.shape,testMsk.dtype, type(testMsk))

# Plot example image 
visualize(Sample_image=testImg.permute(1,2,0),Sample_mask=testMsk.permute(1,2,0))

model = UNet3Plus()
summary(model,testImg.shape)

log = OrderedDict([
    ('epoch', []),
    ('loss', []),
    ('iou', []),
    ('val_loss', []),
    ('val_iou', []),
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer_1 = optim.Adam(params=model.parameters(), lr=0.001)

#criterion = nn.BCELoss()
#criterion = nn.BCEWithLogitsLoss()
criterion = FocalLoss()
epochs=3

model_path = 'C:/Users/Saharsh/Desktop/Onward/Work/medical_image_segmentation/pycache/B_unet3+_model'
log_path = 'C:/Users/Saharsh/Desktop/Onward/Work/medical_image_segmentation/pycache/unet3+_model_logs.csv'

best_iou = 0
trigger = 0

for epoch in range(epochs):
    print(f'Epoch [{epoch}/{epochs}]')

    # train for one epoch
    train_log = train(False, train_data_loader, model, criterion, optimizer_1)
    # evaluate on validation set
    val_log = validate(False, test_data_loader, model, criterion)

    print('loss %.4f , iou %.4f , val_loss %.4f , val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

    log['epoch'].append(epoch)
    log['loss'].append(train_log['loss'])
    log['iou'].append(train_log['iou'])  
    log['val_loss'].append(val_log['loss'])
    log['val_iou'].append(val_log['iou'])

    pd.DataFrame(log).to_csv(log_path, index=False)

    trigger += 1

    if val_log['iou'] > best_iou:
        torch.save(model.state_dict(), model_path)
        best_iou = val_log['iou']
        print("=> saved best model")
        trigger = 0
