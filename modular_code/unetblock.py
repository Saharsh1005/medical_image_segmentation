# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 17:52:18 2022

@author: Saharsh
"""

# LAYERS
import torch
import torch.nn as nn 
from weight_initialize import init_weights

class UnetConv2(nn.Module):
    def __init__(self,in_size,out_size,is_batchnorm, n=2,ks=3,stride=1,padding=1):
      '''
          in_size:      channels of the input image
          out_size:     Total no. of channels after convolution. Each filter processes the input with its own, 
                        different set of kernels and a scalar bias with the process described above,
                        producing a single output channel.
          is_batchnorm: batch normalisation step flag
          ks:           kernel size (3,3)
          stride:       kernel's stride while traversing
          padding:      1 (ensure output H,W same)    
          n:            Number of convolution operations in each unetConv2 step
      '''
      super(UnetConv2,self).__init__()
      self.n = n
      self.ks = ks
      self.stride = stride
      self.padding = padding

      s = stride
      p = padding

      if is_batchnorm:
          for i in range(1, n + 1):
            conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p), nn.BatchNorm2d(out_size), nn.ReLU(inplace=True),)
            setattr(self, 'conv%d' % i, conv)
            in_size = out_size
      else:
          for i in range(1, n + 1):
              conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                    nn.ReLU(inplace=True), )
              setattr(self, 'conv%d' % i, conv)
              in_size = out_size

      '''
      Weight initialization 
      -to overcome vanaishing gradient (gradient->0, when wt. initialise v small )
       and exploding gradient problem (gradient -> infi, when wt initialise v big))
      '''
      for m in self.children():
          init_weights(m, init_type='kaiming') #https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)
        return x

