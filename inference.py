import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data.augment import RandomCrop
from torchvision import transforms
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data.dataloader_npy_external import kneeDataSetSITK

from model_forallplane_MMAL import MRNetResnet_3plane_attslice_MMAL

from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve
from config.mypathlib import model_weight_path, internal_path, nanhai_path, dataset3_path, ExpRoot
from ci.generate_result import gather_result
import warnings
from config.options import TASKLIST

warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_name', default='mrnet_ROIc2', type=str, required=False)
    parser.add_argument('--dataset', default='kneeData', type=str, required=False)
    args = parser.parse_args()
    print(args)
    return args


INFERENCEPATH = ExpRoot
# DATASET_PATH = nanhai_path
DATASET_PATH = internal_path
suffix_time = datetime.now().strftime("%m%d")

if not os.path.exists(INFERENCEPATH):
    os.makedirs(INFERENCEPATH)

def summary(labelsVa, pred):
    labelsVa, pred = np.array(labelsVa), np.array(pred)
    auc = roc_auc_score(labelsVa, pred)
    TP = np.sum((labelsVa == 1) * (pred >= 0.5))
    FN = np.sum((labelsVa == 1) * (pred < 0.5))
    FP = np.sum((labelsVa == 0) * (pred >= 0.5))
    TN = np.sum((labelsVa == 0) * (pred < 0.5))
    TPR = TP / (TP + FN)
    SPC = TN / (TN + FP)
    Acc = (TP + TN) / (TP + FN + FP + TN)
    print(f'AUC:{auc}')
    print(f'TP:{TP}\t FN:{FN}')
    print(f'FP:{FP}\t TN:{TN}')
    print(f'Specificity:{SPC}')
    print(f'Sensitivity:{TPR}')
    print(f'Accuracy:{Acc}')

def evaluate_model(model, val_loader, mode = 'test'):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    outputs = []
    y_trues = []
    y_preds = []
    y_preds_2 = []
    losses = []
    tbar = tqdm(val_loader)
    for i, (images, label, weight, name) in enumerate(tbar):
        if name == -2:
            print(name)
            continue
        # print(_)
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        with torch.no_grad():
            prediction, pred_sagPDW_output, pred_corPDW_output, pred_axiPDW_output, \
                local_prediction, local_pred_sagPDW, local_pred_corPDW, local_pred_axiPDW \
                    = model.forward(images)
        loss = F.cross_entropy(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.softmax(prediction, dim=1)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0][1].item())

        y_preds_2.append(int(probas[0][1].item() > 0.5))
        outputs.append([name[0], probas[0][1].item()])

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        acc = metrics.accuracy_score(y_trues, y_preds_2)

        tbar.set_description('Val/Test loss: %.3f, acc: %.3f, auc: %.3f' % (np.mean(losses),
                                                                            acc,
                                                                            auc))  #


    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    print(mode + ' metrics ********************************************')
    summary(y_trues, y_preds)
    return val_loss_epoch, val_auc_epoch, outputs

def run(args):
    task_list = TASKLIST
    MODELSPATH = model_weight_path

    for task in task_list:
        for checkpoint in os.listdir(MODELSPATH):
            if task in checkpoint:
                modelPth = checkpoint
                break

        print(modelPth)
        if args.dataset == 'kneeData':
            # task = args.task
            task2class = {task: i for i, task in enumerate(task_list)}
            label = task2class[task]

            test_dataset = kneeDataSetSITK('val', dataset_path = DATASET_PATH, label=label, transform=None, sampleP=False, centerCrop=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=1, shuffle=False, drop_last=False)

        model = torch.load(os.path.join(MODELSPATH, modelPth))

        if torch.cuda.is_available():
            model = model.cuda()

            val_loss = 0.0
            val_auc = 0.0

            te_loss, te_auc, te_pred = evaluate_model(model, test_loader, mode='test')
            print(len(te_pred))

            print("task:{0} val loss : {1} | val auc {2} | test loss {3} | test auc {4} ".format(
                            task, val_loss, val_auc, te_loss, te_auc))

            pred_saved_path = f'{INFERENCEPATH}/{args.prefix_name}_{task}'

            te_pred_saved = pred_saved_path+f'_test_{suffix_time}.csv' #os.path.join(pred_saved_path, 'test.csv')
            te_pred = pd.DataFrame(te_pred)
            te_pred.to_csv(te_pred_saved, index=False)

    gather_result(path=INFERENCEPATH, outputname=f'{args.prefix_name}_all_test_{suffix_time}.csv')
if __name__ == "__main__":
    args = parse_arguments()
    run(args)