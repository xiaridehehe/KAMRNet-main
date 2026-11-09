import re
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import scipy.io as sio
import torchvision.transforms as tr
import random
import pandas as pd
from scipy.ndimage import zoom
import math
import random

def resize(image, scale=1):
    output_size = image.shape
    zoom_sq = [1, scale, scale]
    resize_img = zoom(image, zoom=zoom_sq)
    scale_size = resize_img.shape
    center_coord = [scale_size[0]/2, scale_size[1]/2, scale_size[2]/2, ]
    output_image = resize_img[int(center_coord[0]-output_size[0]/2) : int(center_coord[0]+output_size[0]/2),
                            int(center_coord[1]-output_size[1]/2) : int(center_coord[1]+output_size[1]/2),
                            int(center_coord[2]-output_size[2]/2) : int(center_coord[2]+output_size[2]/2)]

    return output_image

def RondomFlip(image, p=0.5, target=None):
    h_v = random.random()
    image = image.numpy()
    if target is not None:
        target = target.numpy()
    # horizontal flip with p = self.p
    if h_v > 0.5:
        if random.random() < p:
            image = image.swapaxes(2, 0)
            image = image[::-1, ...]
            image = image.swapaxes(0, 2)
            if target is not None:
                target = target.swapaxes(2, 0)
                target = target[::-1, ...]
                target = target.swapaxes(0, 2)
    # vertical flip with p = self.p
    else:
        if random.random() < p:
            image = image.swapaxes(1, 0)
            image = image[::-1, ...]
            image = image.swapaxes(0, 1)
            if target is not None:
                target = target.swapaxes(1, 0)
                target = target[::-1, ...]
                target = target.swapaxes(0, 1)
    if target is None:
        # must copy because torch doesnt current support neg strides
        return torch.from_numpy(image.copy()), None
    else:
        return torch.from_numpy(image.copy()),torch.from_numpy(target.copy())


def RandomTranslate(image, translation_range, target=None):
    # height shift
    height_range = translation_range[0]
    width_range = translation_range[1]
    random_height = random.uniform(-height_range, height_range)
    # width shift
    random_width = random.uniform(-width_range, width_range)
    tx = random_height * image.size(1)
    ty = random_width * image.size(2)
    translation_matrix = torch.FloatTensor([[1, 0, tx],
                                         [0, 1, ty],
                                         [0, 0, 1]])
    if target is not None:
        target = th_affine2d(target,
                            translation_matrix,
                            mode='nearest',
                            center=True)
    image = th_affine2d(image,
                           translation_matrix,
                           mode='bilinear',
                           center=True)
    return image, target


def th_iterproduct(*args):
    return torch.from_numpy(np.indices(args).reshape((len(args),-1)).T)

def th_affine2d(x, matrix, mode='bilinear', center=True):
    """
    2D Affine image transform on th.Tensor

    Arguments
    ---------
    x : th.Tensor of size (C, H, W)
        image tensor to be transformed

    matrix : th.Tensor of size (3, 3) or (2, 3)
        transformation matrix

    mode : string in {'nearest', 'bilinear'}
        interpolation scheme to use

    center : boolean
        whether to alter the bias of the transform
        so the transform is applied about the center
        of the image rather than the origin

    Example
    -------
    # >>> import torch
    # >>> from torchsample.utils import *
    # >>> x = th.zeros(2,1000,1000)
    # >>> x[:,100:1500,100:500] = 10
    # >>> matrix = th.FloatTensor([[1.,0,-50],
    # ...                             [0,1.,-50]])
    # >>> xn = th_affine2d(x, matrix, mode='nearest')
    # >>> xb = th_affine2d(x, matrix, mode='bilinear')
    """

    if matrix.dim() == 2:
        matrix = matrix[:2, :]
        matrix = matrix.unsqueeze(0)
    elif matrix.dim() == 3:
        if matrix.size()[1:] == (3, 3):
            matrix = matrix[:, :2, :]

    A_batch = matrix[:, :, :2]
    if A_batch.size(0) != x.size(0):
        A_batch = A_batch.repeat(x.size(0), 1, 1)
    b_batch = matrix[:, :, 2].unsqueeze(1)

    # make a meshgrid of normal coordinates
    _coords = th_iterproduct(x.size(1), x.size(2))
    coords = _coords.unsqueeze(0).repeat(x.size(0), 1, 1).float()

    if center:
        # shift the coordinates so center is the origin
        coords[:, :, 0] = coords[:, :, 0] - (x.size(1) / 2. - 0.5)
        coords[:, :, 1] = coords[:, :, 1] - (x.size(2) / 2. - 0.5)
    # apply the coordinate transformation
    new_coords = coords.bmm(A_batch.transpose(1, 2)) + b_batch.expand_as(coords)

    if center:
        # shift the coordinates back so origin is origin
        new_coords[:, :, 0] = new_coords[:, :, 0] + (x.size(1) / 2. - 0.5)
        new_coords[:, :, 1] = new_coords[:, :, 1] + (x.size(2) / 2. - 0.5)

    # map new coordinates using bilinear interpolation
    if mode == 'nearest':
        x_transformed = th_nearest_interp2d(x.contiguous(), new_coords)
    elif mode == 'bilinear':
        x_transformed = th_bilinear_interp2d(x.contiguous(), new_coords)

    return x_transformed


def th_nearest_interp2d(input, coords):
    """
    2d nearest neighbor interpolation th.Tensor
    """
    # take clamp of coords so they're in the image bounds
    x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 1).round()
    y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 1).round()

    stride = torch.LongTensor(input.stride())
    x_ix = x.mul(stride[1]).long()
    y_ix = y.mul(stride[2]).long()

    input_flat = input.view(input.size(0), -1)

    mapped_vals = input_flat.gather(1, x_ix.add(y_ix))

    return mapped_vals.view_as(input)


def th_bilinear_interp2d(input, coords):
    """
    bilinear interpolation in 2d
    """
    x = torch.clamp(coords[:, :, 0], 0, input.size(1) - 2)
    x0 = x.floor()
    x1 = x0 + 1
    y = torch.clamp(coords[:, :, 1], 0, input.size(2) - 2)
    y0 = y.floor()
    y1 = y0 + 1

    stride = torch.LongTensor(input.stride())
    x0_ix = x0.mul(stride[1]).long()
    x1_ix = x1.mul(stride[1]).long()
    y0_ix = y0.mul(stride[2]).long()
    y1_ix = y1.mul(stride[2]).long()

    input_flat = input.view(input.size(0), -1)

    vals_00 = input_flat.gather(1, x0_ix.add(y0_ix))
    vals_10 = input_flat.gather(1, x1_ix.add(y0_ix))
    vals_01 = input_flat.gather(1, x0_ix.add(y1_ix))
    vals_11 = input_flat.gather(1, x1_ix.add(y1_ix))

    xd = x - x0
    yd = y - y0
    xm = 1 - xd
    ym = 1 - yd

    x_mapped = (vals_00.mul(xm).mul(ym) +
                vals_10.mul(xd).mul(ym) +
                vals_01.mul(xm).mul(yd) +
                vals_11.mul(xd).mul(yd))

    return x_mapped.view_as(input)

import scipy.ndimage.interpolation as scy
def RandomRotate(image, rotation_range, target=None):
    angle = random.uniform(-rotation_range, rotation_range)
    # random.shuffle(angle_list)

    theta = math.pi / 180 * angle
    rotation_matrix = torch.FloatTensor([[math.cos(theta), -math.sin(theta), 0],
                                      [math.sin(theta), math.cos(theta), 0],
                                      [0, 0, 1]])

    if target is not None:
        target = th_affine2d(target,
                            rotation_matrix,
                            mode='nearest',
                            center=True)
    image = th_affine2d(image,
                           rotation_matrix,
                           mode='bilinear',
                           center=True)

    return image, target