# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:21:55 2022

@author: Saharsh
"""
import os
import cv2
import numpy as np
import torch.utils.data
from typing import List

class NuclearSegmentationDataset(torch.utils.data.Dataset):
  def __init__(self,imageList: List,maskList: List,transform= None,):
      self.imageList = imageList
      self.maskList = maskList
      self.transform=None

  def __len__(self):
    return len(self.imageList)

  def __getitem__(self,idx):
      image = self.imageList[idx]
      mask = self.maskList[idx]
      image = image.astype('uint8')
      mask = mask/255

      if (self.transform is not None):
         transformation = self.transform(image,mask)
         image=transformation['image'] 
         mask=transformation['mask']
     
      image=torch.from_numpy(image)
      mask=torch.from_numpy(mask)
      image=image.permute(2,0,1)
      mask=mask.permute(2,0,1)
      image = image.float()/255
      #print("mask:\n",mask)
 
      return image,mask
