import shutil
import os
import time
from datetime import datetime
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from data.augment import RandomCrop
from torchvision import transforms
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, ToTensor, Compose, RandomAffine, RandomResizedCrop, ColorJitter
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from data.dataloader import kneeDataSetSITK

from model_forallplane_MMAL import MRNetResnet_3plane_attslice_MMAL
from sklearn import metrics
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODELSPATH = ''
checkpoint_path = ''

def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-t', '--task', type=str, required=True,
    #                     choices=['abnormal', 'acl', 'meniscus'])
    # parser.add_argument('-p', '--plane', type=str, required=True,
    #                     choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('-t', '--task', type=str, default='MENI',
                        choices=['MENI', 'ACL', 'CART', 'PCL', 'MCL', 'LCL', 'EFFU', 'CONT', 'PLICA', 'CYST', 'IFP', 'PR'])
    parser.add_argument('-p', '--plane', type=str, default='axiPDW',
                        choices=['sagPDW', 'corPDW', 'corT1WI', 'sagT2WI', 'axiPDW'])
    parser.add_argument('--prefix_name', type=str, default='mrnet_ROIc2')
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='None', choices=['plateau', 'step', 'None'])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=150)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args



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


def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every=100):
    _ = model.train()

    if torch.cuda.is_available():
        model.cuda()

    y_preds = []
    y_trues = []
    y_preds_2 = []
    losses = []
    tbar = tqdm(train_loader)
    count = 0
    for i, (images, label, weight, _) in enumerate(tbar):
        if _ == -2:
            count +=1
            continue
        optimizer.zero_grad()

        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        prediction, pred_sagPDW_output, pred_corPDW_output, pred_axiPDW_output, \
            local_prediction, local_pred_sagPDW, local_pred_corPDW, local_pred_axiPDW \
                = model.forward(images)
        
        loss_all = F.cross_entropy(prediction, label.long())
        loss_sagPDW = F.cross_entropy(pred_sagPDW_output, label.long())
        loss_corPDW = F.cross_entropy(pred_corPDW_output, label.long())
        loss_axiPDW = F.cross_entropy(pred_axiPDW_output, label.long())

        loss_all_local = F.cross_entropy(local_prediction, label.long())
        loss_sagPDW_local = F.cross_entropy(local_pred_sagPDW, label.long())
        loss_corPDW_local = F.cross_entropy(local_pred_corPDW, label.long())
        loss_axiPDW_local = F.cross_entropy(local_pred_axiPDW, label.long())

        w_aux = 1.0
        w_local = 0.5
        
        loss = loss_all + w_aux * (loss_sagPDW + loss_corPDW + loss_axiPDW) + \
            w_local * (loss_all_local + loss_sagPDW_local + loss_corPDW_local + \
                       loss_axiPDW_local)


        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)


        probas = torch.softmax(prediction, dim=1)

        # For loss logits
        # y_trues.append(int(label[0][1]))
        # For loss Cross Entropy
        y_trues.append(int(label[0]))
        # y_trues = label.cpu().detach().numpy()
        y_preds.append(probas[0][1].item())
        y_preds_2.append(int(probas[0][1].item() > 0.5))
        # y_preds = probas[:,1].cpu().detach().numpy()

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        # try:
        acc = metrics.accuracy_score(y_trues, y_preds_2)
        #

        # print(auc)
        writer.add_scalar('Train/Loss', loss_value,
                          epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)

        tbar.set_description('Train loss: %.3f, acc: %.3f, auc: %.3f' % (np.mean(losses),
                                                                         acc,
                                                                         auc))  #

    writer.add_scalar('Train/AUC_epoch', auc, epoch + i)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)
    print('Train metrics')
    summary(y_trues, y_preds)
    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, log_every=20):
    _ = model.eval()

    if torch.cuda.is_available():
        model.cuda()

    y_trues = []
    y_preds = []
    y_preds_2 = []
    losses = []
    tbar = tqdm(val_loader)
    for i, (images, label, weight, _) in enumerate(tbar):
        if _ == -2:
            continue
        # print(_)
        if torch.cuda.is_available():
            images = [image.cuda() for image in images]
            label = label.cuda()
            weight = weight.cuda()

        label = label[0]
        weight = weight[0]

        # prediction = model.forward(images)
        with torch.no_grad():
            prediction, pred_sagPDW_output, pred_corPDW_output, pred_axiPDW_output, \
                local_prediction, local_pred_sagPDW, local_pred_corPDW, local_pred_axiPDW \
                    = model.forward(images)
        loss = F.cross_entropy(prediction, label.long())

        loss_value = loss.item()
        losses.append(loss_value)

        # probas = torch.sigmoid(prediction)
        probas = torch.softmax(prediction, dim=1)

        # For loss logits
        # y_trues.append(int(label[0][1]))
        # For loss Cross Entropy
        y_trues.append(int(label[0]))

        y_preds.append(probas[0][1].item())
        y_preds_2.append(int(probas[0][1].item() > 0.5))

        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
        except:
            auc = 0.5

        acc = metrics.accuracy_score(y_trues, y_preds_2)

        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)

        tbar.set_description('Val/Test loss: %.3f, acc: %.3f, auc: %.3f' % (np.mean(losses),
                                                                            acc,
                                                                            auc))  #


    writer.add_scalar('Val/AUC_epoch', auc, epoch + i)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)
    print('Val/Test metrics')
    summary(y_trues, y_preds)
    return val_loss_epoch, val_auc_epoch


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def run(args):
    log_root_folder = "./logs/{0}/".format(args.task)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    augmentor = Compose([
        ToTensor(),
        RandomRotation(25),
        RandomHorizontalFlip(),
        RandomAffine(25),
        RandomResizedCrop(224, scale=(0.8, 1.0)),
    ])

    task = args.task
    task2class = {'MENI': 0, 'ACL': 1, 'CART': 2, 'PCL': 3, 'MCL': 4, 'LCL': 5, 'EFFU': 6, 'CONT': 7, 'PLICA': 8,
                  'CYST': 9, 'IFP': 10, 'PR': 11}
    label = task2class[task]
    plane = args.plane

    train_dataset = kneeDataSetSITK('train', label, transform=augmentor, sampleP=False, centerCrop=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=16, shuffle=True,
                                               drop_last=False)

    validation_dataset = kneeDataSetSITK('val', label, transform=None, sampleP=False, centerCrop=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, num_workers=16, shuffle=True,
                                                    drop_last=False)

    test_dataset = kneeDataSetSITK('test', label, transform=None, sampleP=False, centerCrop=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=16, shuffle=True, drop_last=False)
   
    net = MRNetResnet_3plane_attslice_MMAL()
    

    modelPths = os.listdir(checkpoint_path)
    for model_name in modelPths:
        if args.task in model_name:
            checkpoint = torch.load(checkpoint_path + model_name)
            break
        
    net = checkpoint

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.1)  # origin
    

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=8, factor=.5, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=args.gamma)

    best_val_loss = float('inf')
    best_test_auc = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()

        train_loss, train_auc = train_model(
            net, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_every)
        val_loss, val_auc = evaluate_model(
            net, validation_loader, epoch, num_epochs, writer, current_lr)
        test_loss, test_auc = evaluate_model(
            net, test_loader, epoch, num_epochs, writer, current_lr)

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print(task)
        print(
            "train loss : {0} | train auc {1} | val loss {2} | val auc {3} | test loss{4} | test auc{5} | elapsed time {6} s".format(
                train_loss, train_auc, val_loss, val_auc, test_loss, test_auc, delta))

        iteration_change_loss += 1
        print('-' * 50)

        if test_auc > best_test_auc:
            best_val_auc = val_auc
            if bool(args.save_model):
                file_name = f'modelresnet_{args.prefix_name}_{args.task}_allplane_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_test_auc_{test_auc:0.4f}_epoch_{epoch + 1}.pth'
                torch.save(net, f'{MODELSPATH}/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
