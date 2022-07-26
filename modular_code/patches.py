# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:40:57 2022

@author: Saharsh
"""
from PIL import Image
import cv2
import numpy as np
from patchify import patchify, unpatchify

def create_patches(image_path, patch_size,step_size,read_as_rgb_bw=0):
    '''
    This function is used to split an image or a mask into patches 
    
    image_path:     path of image file
    patch_size:     size of patches to be created
    step_size:      stride while creating patches. if step_size==patch_size then no overlap
    read_as_rgb_bw: 0 - to read as black and white or 1 read as rgb or bgr (flag)
    
    returns a list of patches (list of numpy array)
    
    '''
    all_img_patches = []
    
    image = cv2.imread(image_path, read_as_rgb_bw)  #Read each mask as Gray
    size_x = (image.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    size_y = (image.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    image = Image.fromarray(image)
    image = image.crop((0 ,0, size_x, size_y))  #Crop from top left corner
    image = np.array(image)               
    #Extract patches from each image
    i_name= image_path.split('/')[-1]

    if read_as_rgb_bw == 1:
        image_patches = patchify(image, (patch_size, patch_size,3), step=step_size)  #Step= 256 for 256 patches means no overlap
    else:
        image_patches = patchify(image, (patch_size, patch_size), step=patch_size)
    

    for i in range(image_patches.shape[0]):
        for j in range(image_patches.shape[1]):              
            single_patch_image = image_patches[i,j,:,:]  
            if read_as_rgb_bw == 1:
                single_patch_image = single_patch_image[0]
            else:
                pass
            
            all_img_patches.append(single_patch_image)
            
    return all_img_patches
         
    
    