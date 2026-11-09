import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import torch.utils.data as data
import glob
import torchvision.transforms as transforms
from datatransform import *

from torch.autograd import Variable
from data import medical_image_process as img_loader
from skimage import transform
import SimpleITK as sitk
import warnings
import pandas as pd
import random
import json
import math

INPUT_DIM = 224
MAX_PIXEL_VAL = 255
MEAN = 58.09
STDDEV = 49.73


class Dataset(data.Dataset):
    def __init__(self, datadirs, diagnosis, use_gpu, transform=None):
        super().__init__()
        self.use_gpu = use_gpu
        self.transform = transform

        label_dict = {}
        self.paths = []

        for i, line in enumerate(open('metadata.csv').readlines()):
            if i == 0:
                continue
            line = line.strip().split(',')
            path = line[10]
            label = line[2]
            label_dict[path] = int(int(label) > diagnosis)

        for dir in datadirs:
            for file in os.listdir(dir):
                self.paths.append(dir + '/' + file)

        self.labels = [label_dict[path[6:]] for path in self.paths]

        neg_weight = np.mean(self.labels)
        self.weights = [neg_weight, 1 - neg_weight]

    def weighted_loss(self, prediction, target):
        weights_npy = np.array([self.weights[int(t[0])] for t in target.data])
        weights_tensor = torch.FloatTensor(weights_npy)
        if self.use_gpu:
            weights_tensor = weights_tensor.cuda()
        loss = F.binary_cross_entropy_with_logits(prediction, target, weight=Variable(weights_tensor))
        return loss

    def __getitem__(self, index):
        path = self.paths[index]
        with open(path, 'rb') as file_handler:  # Must use 'rb' as the data is binary
            vol = pickle.load(file_handler).astype(np.int32)

        # crop middle
        pad = int((vol.shape[2] - INPUT_DIM) / 2)
        vol = vol[:, pad:-pad, pad:-pad]

        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
        # normalize
        vol = (vol - MEAN) / STDDEV
        # convert to RGB
        vol = np.stack((vol,) * 3, axis=1)

        vol_tensor = torch.FloatTensor(vol)
        label_tensor = torch.FloatTensor([self.labels[index]])

        return vol_tensor, label_tensor

    def __len__(self):
        return len(self.paths)


def readLabels(labelPath):
    import pandas as pd
    sampleIDWithLabel = []
    labelData = pd.read_csv(labelPath)
    for i in range(len(labelData)):
        d = list(labelData.iloc[i, :])
        sampleIDWithLabel.append(d)
    return sampleIDWithLabel


class kneeDataSet(data.Dataset):
    def __init__(self, mode, dataset_path='./datasets', classes=2, transform=None):
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        self.training_path = self.root + '/kneeData/training'
        self.label_path = self.root + '/kneeData/labels'
        list_reg_t1 = sorted(glob.glob(os.path.join(self.training_path, '*/MR2Nii/sag PDW.nii')))
        self.list = self.create_cls_dataset(list_reg_t1, root=self.training_path,
                                            labelCSVPath=os.path.join(self.label_path, 'train.csv'))
        self.list_test = self.create_cls_dataset(list_reg_t1, root=self.training_path,
                                                 labelCSVPath=os.path.join(self.label_path, 'test.csv'))
        self.transform = transform

        if mode == 'val':
            split_id = list(range(0, int(0.2 * len(self.list))))
            self.list = [self.list[i] for i in split_id]
        elif mode == 'train':
            split_id = list(range(int(0.2 * len(self.list)), len(self.list)))
            self.list = [self.list[i] for i in split_id]
        else:
            self.list = self.list_test

    def weighted_loss(self, prediction, target):
        loss = F.binary_cross_entropy_with_logits(prediction, target)
        return loss

    def create_cls_dataset(self, *ls, root, labelCSVPath):
        total = len(ls[0])
        print(ls[0])
        assert total != 0, "Problem reading data. Check the data paths."
        labels = readLabels(labelCSVPath)
        list = []
        for label in labels:
            label[0] = os.path.join(root, label[0], 'MR2Nii', 'sag PDW.nii')
            list.append(label)
        return list


    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        imgPath, labels = self.list[index][0], self.list[index][1:2]
        t1_img = img_loader.load_medical_image(imgPath, type="T1", normalization='max_min')
        t1_img = transform.resize(t1_img, [320, 320, 48])
        # print(t1_img.shape)
        t1_img = np.transpose(t1_img, (2, 0, 1))
        # print(t1_img.shape)
        vol = t1_img
        # standardize
        vol = (vol - np.min(vol)) / (np.max(vol) - np.min(vol)) * MAX_PIXEL_VAL
        # normalize
        vol = (vol - MEAN) / STDDEV
        # convert to RGB
        vol = np.stack((vol,) * 3, axis=1)
        label_tensor = torch.FloatTensor(labels)
        vol_tensor = torch.FloatTensor(vol)

        return imgPath, vol_tensor, label_tensor


def padding(A, ps=(240, 240)):
    d, w, h = A.shape
    pw, ph = ps
    wp1 = int((pw - w) / 2)
    if pw - w > 0:
        wp2 = wp1 + (pw - w) % 2
    else:
        wp2 = wp1 - (pw - w) % 2
    hp1 = int((ph - h) / 2)
    if ph - h > 0:
        hp2 = hp1 + (ph - h) % 2
    else:
        hp2 = hp1 - (ph - h) % 2

    if hp1 < 0:
        if hp2 < 0:
            A = A[:, :, -hp1:hp2]
        else:
            A = A[:, :, -hp1:-hp2]
    else:
        if hp2 < 0:
            A = A[:, :, hp1:hp2]
        else:
            A = np.pad(A, ((0, 0), (0, 0), (hp1, hp2)))
    if wp1 < 0:
        if wp2 < 0:
            A = A[:, -wp1:wp2, :]
        else:
            A = A[:, -wp1:-wp2, :]
    else:
        if wp2 < 0:
            A = A[:, wp1:wp2, :]
        else:
            A = np.pad(A, ((0, 0), (wp1, wp2), (0, 0)))

    return A


class kneeDataSetSITK(data.Dataset):
    def __init__(self, mode, label, dataset_path, classes=2, transform=None,
                 sampleP=True, centerCrop=False):
        self.label = label
        self.mode = mode
        self.root = dataset_path
        self.classes = classes
        self.transform = transform
        self.sampleP = sampleP
        self.centerCrop = centerCrop

        self.training_path = self.root + '/data_npy_nor'
        print(self.training_path)
        self.label_path = self.root
        self.ROI_path = dataset_path.roi

        with open(self.ROI_path, 'r') as f:
            self.centerROI = json.load(f)
        self.className = ['MENI', 'ACL', 'CART', 'PCL', 'MCL', 'LCL', 'EFFU', 'CONT', 'PLICA', 'CYST', 'IFP', 'PR'][label]

        if mode == 'train':
            self.list_train = self.create_cls_dataset(root=self.training_path,
                                                      labelCSVPath=dataset_path.train_label)
            self.list_org = self.list_train
        elif mode == 'val':
            self.list_val = self.create_cls_dataset(root=self.training_path,
                                                    labelCSVPath=dataset_path.val_label)
            self.list_org = self.list_val
        else:
            self.list_test = self.create_cls_dataset(root=self.training_path,
                                                     labelCSVPath=dataset_path.test_label)
            self.list_org = self.list_test


        pos = np.sum([item[self.label + 1] for item in self.list_org])
        neg = len(self.list_org) - pos
        self.negrate = neg * 1.0 / len(self.list_org)
        self.posrate = pos * 1.0 / len(self.list_org)
        print(f'neg:{neg}, pos:{pos}')
        self.negvspos = neg * 1.0 / (pos * 1.0)
        self.posvsneg = pos * 1.0 / (neg * 1.0)
        # print(f'pos:{posrate}, neg:{negrate}')
        self.weights = torch.FloatTensor([1, neg * 1.0 / pos])

        self.list = []


        self.posvsneg = 0.5
        self.negvspos = 2
        if mode == 'train':
            count = 0
            if self.negvspos > 1:
                for data_pat in self.list_org:
                    if data_pat[self.label+1] == 1:
                        for i in range(int(max(1, self.negvspos))):
                            self.list.append(data_pat)
                    else:
                        self.list.append(data_pat)
                random.shuffle(self.list)
            else:
                for data_pat in self.list_org:
                    if data_pat[self.label + 1] == 0:
                        for i in range(int(max(1, self.posvsneg))):
                            self.list.append(data_pat)
                    else:
                        self.list.append(data_pat)
                random.shuffle(self.list)
        else:
            self.list = self.list_org

        random.shuffle(self.list)

        pos = np.sum([item[self.label + 1] for item in self.list])
        neg = len(self.list) - pos
        self.negrate = neg * 1.0 / len(self.list)
        self.posrate = pos * 1.0 / len(self.list)
        print(f'new neg:{neg}, pos:{pos}')

        if neg < pos:
            self.posGreaterNeg = True
        else:
            self.posGreaterNeg = False

    def create_cls_dataset(self, root, labelCSVPath):
        labels = readLabels(labelCSVPath)
        list = []
        for label in labels:
            # sitk
            label.append(label[0])
            label[0] = os.path.join(root, label[0]+'.npy')

            list.append(label)
        return list


    def __len__(self):
        return len(self.list)

    def sitkReadZoom(self, path, newSize=None, centerCrop=None):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(dicom_names)
        sitk_img = reader.Execute()


        if centerCrop:
            x, y, z = sitk_img.GetSize()
            outputSpacing, size, outputOrigin, outputDirection = sitk_img.GetSpacing(), (
            int(x / 2), int(y / 2), z), sitk_img.GetOrigin(), sitk_img.GetDirection()
            img = sitk.GetArrayFromImage(sitk_img)
            cropX, cropY = int(x / 4), int(y / 4)
            img = img[:, cropX:-cropX, cropY:-cropY]

            sitk_img = sitk.GetImageFromArray(img)
            sitk_img.SetSpacing(outputSpacing)
            sitk_img.SetDirection(outputDirection)

        if newSize:
            newX, newY, newZ = newSize
            dx, dy, dz = sitk_img.GetSpacing()
            x, y, z = sitk_img.GetSize()
            if newZ == -1:
                newZ = z
            resample = sitk.ResampleImageFilter()
            resample.SetOutputSpacing(((x * dx / newX), (y * dy / newY), (z * dz) / newZ))
            resample.SetSize((newX, newY, newZ))
            resample.SetOutputOrigin(sitk_img.GetOrigin())
            resample.SetOutputDirection(sitk_img.GetDirection())
            newImage = resample.Execute(sitk_img)
            # no nuyl
            newImage = sitk.Cast(sitk.RescaleIntensity(newImage), sitk.sitkUInt8)  # nyul
            
            out_array = sitk.GetArrayFromImage(newImage)
            out_array = out_array/255 

            return out_array
        else:
            # no nuyl
            sitk_img = sitk.Cast(sitk.RescaleIntensity(sitk_img), sitk.sitkUInt8)  # nyul
            out_array = sitk.GetArrayFromImage(sitk_img)
            return out_array

    def whiting(self, img):
        img2 = (img - MEAN) / STDDEV
        img2 = ((img2 - np.min(img2)) / (np.max(img2) - np.min(img2)) * MAX_PIXEL_VAL)
        return img2.astype(np.uint8)

    def __getitem__(self, index):
        imgPaths, label, name = self.list[index][0], self.list[index][self.label + 1], self.list[index][-1]
        if self.posGreaterNeg:
            if label == 0:
                label = torch.FloatTensor([0])
            else:
                if self.sampleP:
                    flag = np.random.choice([1, 0], 1, replace=False,
                                            p=[self.negrate / self.posrate, 1 - self.negrate / self.posrate])
                    if flag == 1:
                        label = torch.FloatTensor([1])

                    else:
                        return -2, -2, -2, -2
                else:
                    label = torch.FloatTensor([1])

        else:
            if label == 1:
                label = torch.FloatTensor([1])
            else:
                if self.sampleP:
                    flag = np.random.choice([1, 0], 1, replace=False,
                                            p=[self.posrate / self.negrate, 1 - self.posrate / self.negrate])
                    if flag == 1:
                        label = torch.FloatTensor([0])
                    else:
                        return -2, -2, -2, -2
                else:
                    label = torch.FloatTensor([0])

        label_tensor = label

        try:
            vols = np.load(imgPaths)
        except:
            print('read img error')
            return -2, -2, -2, -2

        vols_new = []
        for vol in vols:
            if self.transform:
                try:
                    vol = self.transform(vol)
                except:
                    print('transform error')
                    return -2, -2, -2, -2
            vol = np.stack((vol,) * 3, axis=1)
            vol = torch.FloatTensor(vol)
            vols_new.append(vol)

        return vols_new, label_tensor, self.weights, name  #


