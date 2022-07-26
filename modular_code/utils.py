# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:33:24 2022

@author: Saharsh
"""
from matplotlib import pyplot as plt

def visualize(**images):
    n = len(images)
    plt.figure(figsize=(12, 12))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        try: #For image with 3 channels
          plt.imshow(image)
        except: #For binary mask
          plt.imshow(image[:,:,0])
    plt.show()
