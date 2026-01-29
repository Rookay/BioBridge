# -*- coding: UTF-8 -*-
# @Project ：BioEEG-VisDec 
# @File    ：Blur.py
# @IDE     ：PyCharm 
# @Date    ：2025/11/28 13:31
import torch
import torch.nn as nn
import kornia
import torch.nn.functional as F

class RadialBlurMultiCurve(nn.Module):

    def __init__(self, H=224, W=224, sigma=6, mode='logistic'):
        super().__init__()

        self.sigma = sigma
        self.mode = mode

        self.k = nn.Parameter(torch.tensor(8.0))          
        self.center = nn.Parameter(torch.tensor(0.3))     
        self.lambda_exp = nn.Parameter(torch.tensor(2.0)) 
        self.gamma_quad = nn.Parameter(torch.tensor(2.0)) 

        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W),
            indexing="ij"
        )
        r = torch.sqrt(xx**2 + yy**2)
        self.register_buffer("radius", r)

    def logistic_weight(self):
        return torch.sigmoid(self.k * (self.radius - self.center))

    def exp_weight(self):
        r = torch.clamp(self.radius - self.center, min=0)
        w = 1 - torch.exp(-torch.clamp(self.lambda_exp, 0.1, 10.0) * r)
        return torch.clamp(w, 0, 1)

    def quad_weight(self):
        r = torch.clamp(self.radius - self.center, min=0)
        gamma = torch.clamp(self.gamma_quad, 1.0, 4.0)
        w = torch.pow(r, gamma)
        return torch.clamp(w, 0, 1)

    # -----------------------------------------
    def forward(self, img):

        blur = kornia.filters.gaussian_blur2d(img, (5,5), (self.sigma, self.sigma))

        if self.mode == 'logistic':
            w = self.logistic_weight()

        elif self.mode == 'exp':
            w = self.exp_weight()

        elif self.mode == 'quad':
            w = self.quad_weight()


        w = w.unsqueeze(0).unsqueeze(0)  
        return img * (1 - w) + blur * w

class DirectT(nn.Module):
    def __init__(self, H=224, W=224):
        super().__init__()
        self.H = H
        self.W = W

    def forward(self, img):

        return img


