from torchvision import models, transforms

import numpy as np
import cv2
import requests
from PIL import Image
from torch.autograd import Variable
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate
import SimpleITK as sitk
from data.dataloader_npy_external import kneeDataSetSITK
import model_forallplane_MMAL
from tqdm import tqdm
import torch.nn as nn
import os
import math
from config.mypathlib import model_weight_path, nanhai_path, internal_path, ExpRoot
from config.options import TASKLIST, TaskToIndx
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

plane_index = {'sagPDW':0, 'corPDW':1, 'axiPDW':2}

MODELSPATH = model_weight_path
ds_path = nanhai_path
outputPath = ExpRoot

selected_name_dict = {}



def feature_hook(model, input, output):
    feature_data.append(output.data.cpu().numpy())


def makeCAM(feature, weights, classes_id):
    # batchsize, C, h, w
    nc, h, w = feature.shape
    # (512,) @ (512, 7*7) = (49,)
    cam = weights[classes_id].dot(feature.reshape(nc, h * w))
    cam = cam.reshape(h, w)  # (7, 7)
    cam = np.exp(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam_gray = np.uint8(255 * cam)
    return cv2.resize(cam_gray, (224, 224))


for task_n_key, selected_name in selected_name_dict.items():
    
    if selected_name == []:
        continue
    task = task_n_key.split('_')[0]
    for checkpoint in os.listdir(MODELSPATH):
        if task in checkpoint:
            modelPth = checkpoint
            break

    test_dataset = kneeDataSetSITK('test', dataset_path=ds_path, label=TaskToIndx[task], transform=False, sampleP=False, centerCrop=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=True, drop_last=False)

    net = torch.load(os.path.join(MODELSPATH, modelPth))  
    net.eval()
    for plane, plane_idx in plane_index.items():

        outdir = os.path.join(outputPath, task_n_key, plane)
        os.makedirs(outdir, exist_ok=True)


        feature_data = []
        net._modules.get('model_'+plane).model.model.layer4[1].register_forward_hook(hook=feature_hook)
        fc_weights_resnet18 = net._modules.get('model_'+plane)._modules.get('classifier').weight.data.cpu().numpy()


        if len(selected_name) != 0:
            select_data = [default_collate([test_dataset.get_by_id(name)]) for name in selected_name]
            tbar = tqdm(select_data, desc=f"Processing {task_n_key} {plane}")
        else:
            tbar = tqdm([])
        for i, (images, label, weight, name) in enumerate(tbar):
            if name == -2:
                continue

            if torch.cuda.is_available():
                images = [image.cuda() for image in images]
                label = label.cuda()
                weight = weight.cuda()

            label = label[0]
            weight = weight[0]
            index = label.cpu().numpy()

            y_true = int(label[0])
            prediction, pred_sagPDW_output, pred_corPDW_output, pred_axiPDW_output, \
            local_prediction, local_pred_sagPDW, local_pred_corPDW, local_pred_axiPDW = net.forward(images)
            probas = np.argmax(torch.softmax(prediction, dim=1).data.cpu().numpy(), axis=1)

            y_pred = int(probas[0].item() > 0.5)

            if True: # always save
                for i in range(images[1].size()[1]):
                    cam_gray_resnet18 = makeCAM(feature_data[0][i,:,:,:], fc_weights_resnet18, probas)
                    cam_gray_resnet18 = np.ones_like(cam_gray_resnet18) - cam_gray_resnet18
                    img = (images[plane_index[plane]].cpu().data.numpy())[0,i,0,:,:]
                    [w,h] = img.shape
                    b = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
                    g = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
                    r = ((img-np.min(img))/(np.max(img)-np.min(img))*255).astype(np.uint8)
                    img = cv2.merge([b,g,r])

                    cam_color_resnet18 = cv2.applyColorMap(cv2.resize(cam_gray_resnet18, (w, h)),
                                                        cv2.COLORMAP_JET)
                    img = cv2.resize(img, (w,h))
                    cam_resnet18 = img * 0.8 + cam_color_resnet18 * 0.2
                    cv2.imwrite(os.path.join(outdir, f"{name[0]}_{str(i)}.jpg"), img)
                    cv2.imwrite(os.path.join(outdir, f"{name[0]}_{str(i)}_CAM.jpg"), cam_resnet18)




