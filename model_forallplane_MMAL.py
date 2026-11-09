import torch
import torch.nn as nn

from torchvision import models

import torch.nn.functional as F
import random
import math
import os
from torch.nn.init import kaiming_uniform_, kaiming_normal_
# from seg_decoder import *


import skimage.measure as measure
def AOLM(fms, fm1):
    fms = torch.unsqueeze(fms, dim=0)
    fms = torch.transpose(fms,1,2)
    fm1 = torch.unsqueeze(fm1, dim=0)
    fm1 = torch.transpose(fm1,1,2)

    A = torch.sum(fms, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3, 4], keepdim=True)
    M = (A > a).float()

    A1 = torch.sum(fm1, dim=1, keepdim=True)
    a1 = torch.mean(A1, dim=[2, 3, 4], keepdim=True)
    M1 = (A1 > a1).float()

    # coordinates = []
    # for i, m in enumerate(M):
    mask_np = M.cpu().numpy().reshape(24, 11, 11)
    component_labels = measure.label(mask_np)

    properties = measure.regionprops(component_labels)
    areas = []
    for prop in properties:
        areas.append(prop.area)
    max_idx = areas.index(max(areas))

    intersection = ((component_labels==(max_idx+1)).astype(int) + (M1[0,0,:,:,:].cpu().numpy()==1).astype(int)) ==2
    prop = measure.regionprops(intersection.astype(int))
    if len(prop) == 0:
        bbox = [0, 0, 0, 24, 11, 11]
        print('there is one img no intersection')
    else:
        bbox = prop[0].bbox
    z_lefttop = bbox[0]
    x_lefttop = bbox[1] * 32 - 1
    y_lefttop = bbox[2] * 32 - 1
    z_rightlow = bbox[3]
    x_rightlow = bbox[4] * 32 - 1
    y_rightlow = bbox[5] * 32 - 1
    # for image
    if x_lefttop < 0:
        x_lefttop = 0
    if y_lefttop < 0:
        y_lefttop = 0
    coordinate = [z_lefttop, x_lefttop, y_lefttop, z_rightlow, x_rightlow, y_rightlow]
        # coordinates.append(coordinate)
    return coordinate


class resnet_new(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrain)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se_linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        conv5_b = self.model.layer4[0](x)
        x = self.model.layer4[1](conv5_b)
        fm = x
        x = self.gap(x).view(x.size(0), -1)
        attention2 = self.se_linear(x)
        x = x * attention2.expand_as(x)
        x = torch.max(x, 0, keepdim=True)[0]
        embedding = x
        return fm, embedding, conv5_b

class MRNetResnet_single_MMAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet_new(pretrain=True)

        self.classifier = nn.Linear(512, 2)

    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        fm, embedding, conv5_b = self.model(x)
        batch_size, channel_size, side_size, _ = fm.shape

        output = self.classifier(embedding)

        coordinate = torch.tensor(AOLM(fm.detach(), conv5_b.detach()))
        local_imgs = torch.zeros([batch_size, 3, 256, 256]).cuda()
        [z0, x0, y0, z1, x1, y1] = coordinate
        # print(coordinate)
        for i in range(z0, z1):
            local_imgs[i:i + 1] = F.interpolate(x[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(256, 256),
                                                mode='bilinear', align_corners=True)  # [N, 3, 224, 224]
        local_fm, local_embedding, _ = self.model(local_imgs.detach()) ###detach！！！！
        local_output = self.classifier(local_embedding)

        # torch.mean(x,dim=0)
        return output, local_output, embedding, local_embedding

class MRNetResnet_3plane_attslice_MMAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_sagPDW = MRNetResnet_single_MMAL()
        self.model_corPDW = MRNetResnet_single_MMAL()
        self.model_axiPDW = MRNetResnet_single_MMAL()

        self.classifier = nn.Linear(512*3, 2)

    def forward(self, x):
        x_sagPDW, x_corPDW, x_axiPDW = x

        out_sagPDW, local_out_sagPDW, x_sagPDW, local_x_sagPDW = self.model_sagPDW(x_sagPDW)
        out_corPDW, local_out_corPDW, x_corPDW, local_x_corPDW = self.model_corPDW(x_corPDW)
        out_axiPDW, local_out_axiPDW, x_axiPDW, local_x_axiPDW = self.model_axiPDW(x_axiPDW)

        x_allplane = torch.cat([x_sagPDW, x_corPDW, x_axiPDW], dim=1)#, x_corT1WI, x_sagT2WI
        out = self.classifier(x_allplane)

        local_x_allplane = torch.cat([local_x_sagPDW, local_x_corPDW, local_x_axiPDW], dim=1)# local_x_corT1WI, local_x_sagT2WI,
        local_out = self.classifier(local_x_allplane)

        return out, out_sagPDW, out_corPDW, out_axiPDW, \
            local_out, local_out_sagPDW, local_out_corPDW, local_out_axiPDW# out_corT1WI, out_sagT2WI, local_out_corT1WI, local_out_sagT2WI









