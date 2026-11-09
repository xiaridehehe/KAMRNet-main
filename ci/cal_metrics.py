'''
Helper script to calculate the acc, auc, sen, spec metrics for the model
'''

import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, cohen_kappa_score, recall_score, precision_score

from matplotlib import pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpkg
from rpy2.robjects.vectors import StrVector
import math
import os
#
from multiprocessing.pool import Pool

from config.options import TASKLIST, thresholds
from utils.select_by_id import match_id

pROC = rpkg.importr('pROC')
irr = rpkg.importr('irr')

pandas2ri.activate()



def cal_metrics(task, prob_file, label_file):
    '''
    label_file can be df or str path
    
    '''
    a = pd.read_csv(prob_file)
    if type(label_file) == pd.DataFrame:
        b = label_file
    else:
        b = pd.read_csv(label_file)
    a.sort_values(by='ID', axis=0, inplace=True)
    a.reset_index(drop=True, inplace=True)
    b.sort_values(by='ID', axis=0, inplace=True)
    b.reset_index(drop=True, inplace=True)
    # check if a and b have same first column
    # select the columns in TASKLIST
    b = b[['ID', *TASKLIST]]
    try:
        a = match_id(a, b)
    except AssertionError as e:
        raise AssertionError(prob_file, e)

    csvdata_pred = ro.conversion.py2rpy(a)
    csvdata_gt = ro.conversion.py2rpy(b)
    heads_pred = csvdata_pred.colnames.tolist()
    heads_gt = csvdata_gt.colnames.tolist()

    if task != 'all':
        gt_idx = heads_gt.index(task)
        gt = csvdata_gt[gt_idx]
        pred_idx = heads_pred.index(task)
        pred = csvdata_pred[pred_idx]
    else:
        gt = np.zeros((9 * a.shape[0],))
        pred = np.zeros((9 * a.shape[0],))
        for i in range(1, 10):
            gt[(i - 1) * a.shape[0]: i * a.shape[0]] = csvdata_gt[i]
            pred[(i - 1) * a.shape[0]: i * a.shape[0]] = csvdata_pred[i]

    # get roc object
    roc = pROC.roc(gt, pred)

    sen_list = [round(x, 3) for x in roc[1].tolist()]
    sen_list.reverse()


    # choose by the thresholddd
    spec_best = pROC.coords(roc, x=thresholds[task], input='threshold', ret='specificity', transpose=False)[0]
    sens_best = pROC.coords(roc, x=thresholds[task], input='threshold', ret='sensitivity', transpose=False)[0]
    print(task, roc.rx2('auc')[0])

    se_low_best, se_med_best, se_high_best = pROC.ci(roc, of='se', specificities=spec_best)[0]
    sp_low_best, sp_med_best, sp_high_best = pROC.ci(roc, of='sp', sensitivities=sens_best)[0]

    acc_best = pROC.coords(roc, x=spec_best, input='specificity', ret='accuracy', transpose=False)[0]
    acc_ci_array = pROC.ci(roc, of='coords', x=spec_best, input='specificity', ret='accuracy', transpose=False)[0]
    acc_low_best, acc_med_best, acc_high_best = acc_ci_array[0][0], acc_ci_array[0][1], acc_ci_array[0][2]


    auc_low, auc, auc_high = pROC.ci(roc)

    M = np.concatenate((np.array([pred]), np.array([gt])), axis=0).transpose()
    _, num_subject, raters, irr_name, kappa_score, stats_name, z, p_value = irr.kappa2(M, 'unweighted')
    position_ci = ro.r('positon_ci <- qnorm(c(.025,.975))')
    kappa_low, kappa_high = kappa_score + position_ci * (kappa_score / z)

    stats_dict = {'doc_name': 'AI MODEL',
                  # 'roc': roc,
                  'auc': auc,
                  'auc_low': auc_low,
                #   'auc_med': auc_med,
                  'auc_high': auc_high,
                  'sen': sens_best,
                  'spe': spec_best,
                  'sen_low': se_low_best,
                  # 'sen_med': se_med_best,
                  'sen_high': se_high_best,
                  'spe_low': sp_low_best,
                  # 'spe_med': sp_med_best,
                  'spe_high': sp_high_best,
                  'acc': acc_best,
                  'acc_low': acc_low_best,
                  'acc_high': acc_high_best,
                  'kappa': kappa_score[0],
                  'kappa_low': kappa_low,
                  'kappa_high': kappa_high
                  }
    return  stats_dict

