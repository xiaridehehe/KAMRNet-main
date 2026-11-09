'''
use mcnemar test to calculate p value for doctor results
'''



import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from statsmodels.stats.contingency_tables import  mcnemar

from matplotlib import pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpkg
from rpy2.robjects.vectors import StrVector
import math
import os
from scipy import stats

from multiprocessing.pool import Pool
from config.options import TASKLIST
from utils.select_by_id import match_id

pROC = rpkg.importr('pROC')
irr = rpkg.importr('irr')

pandas2ri.activate()



def process_one(inputs):
    sen, spe, roc = inputs
    se_low, se_med, se_high = pROC.ci(roc, of='se', specificities=spe)[0]
    spe_low, spe_med, spe_high = pROC.ci(roc, of='sp', sensitivities=sen)[0]

    return spe, se_low, se_med, se_high, spe_low, spe_med, spe_high

def run(output_dir, label_file, bool_file, doc_result_path):
    '''
    if boolfile is provided, compare doctor with AI.
    Otherwise, compare doctor, using their result with and without AI.
    '''
    SAVE_PATH = output_dir
    b = pd.read_csv(label_file)
    if bool_file is not None:
        print("ai file is provided, compare doctor with AI")
        compare_ai = True
    else:
        compare_ai = False
    data_path = doc_result_path

    p_result = []
    for head in [ 'junior', 'senior', 'all',]:
        print(head)
        for task in TASKLIST:#
            print(task)
            b.sort_values(by='ID', axis=0, inplace=True)
            b.reset_index(drop=True, inplace=True)
            csvdata_gt = ro.conversion.py2rpy(b)
            heads_gt = csvdata_gt.colnames.tolist()

            preds_comp = np.array([])
            preds_comp_ai = np.array([])
            for doc_result in os.listdir(os.path.join(data_path, 'woAI')):
                print(doc_result)
                doc_name = doc_result.split('.')[0].split("+")[0]  
                if head == 'junior':
                    if doc_name in ['QY', 'YHZ']:
                        continue
                elif head == 'senior':
                    if doc_name not in ['QY', 'YHZ']:
                        continue
                wo_AI_file = os.path.join(data_path, 'woAI', f'{doc_name}.csv')
                w_AI_file = os.path.join(data_path, 'wAI', f'{doc_name}+AI.csv')
                a = pd.read_csv(wo_AI_file)
                if compare_ai:
                    c = pd.read_csv(bool_file)
                else:
                    c = pd.read_csv(w_AI_file)


                a.sort_values(by='ID', axis=0, inplace=True)
                a.reset_index(drop=True, inplace=True)
                c.sort_values(by='ID', axis=0, inplace=True)
                c.reset_index(drop=True, inplace=True)

                b = b[['ID',*TASKLIST]]
                try:
                    a = match_id(a, b)
                except AssertionError as e:
                    raise AssertionError(wo_AI_file, e)
                try:
                    c = match_id(c, b)
                except AssertionError as e:
                    raise AssertionError(w_AI_file, e)


                csvdata_pred = ro.conversion.py2rpy(a)
                csvdata_pred_ai = ro.conversion.py2rpy(c)
                heads_pred = csvdata_pred.colnames.tolist()
                heads_pred_ai = csvdata_pred_ai.colnames.tolist()

                if task!= 'all':
                    pred_idx = heads_pred.index(task)
                    pred = csvdata_pred[pred_idx]
                    pred_idx_ai = heads_pred_ai.index(task)
                    pred_ai = csvdata_pred_ai[pred_idx_ai]
                    gt_idx = heads_gt.index(task)
                    gt = csvdata_gt[gt_idx]

                else:
                    gt = np.zeros(((len(heads_pred) - 1) * c.shape[0],))
                    pred = np.zeros(((len(heads_pred) - 1) * c.shape[0],))
                    pred_ai = np.zeros(((len(heads_pred) - 1) * c.shape[0],))
                    for i in range(1, len(heads_gt)):
                        gt[(i - 1) * c.shape[0]: i * c.shape[0]] = csvdata_gt[i]
                        pred[(i - 1) * c.shape[0]: i * c.shape[0]] = csvdata_pred[i]
                        pred_ai[(i - 1) * c.shape[0]: i * c.shape[0]] = csvdata_pred_ai[i]

                preds_comp = np.concatenate([preds_comp, pred]).ravel()
                preds_comp_ai = np.concatenate([preds_comp_ai, pred_ai]).ravel()
            if head == 'junior':
                gt_comp = np.concatenate([np.array(gt), np.array(gt), np.array(gt), np.array(gt)])

            elif head == 'senior':
                gt_comp = np.concatenate([np.array(gt), np.array(gt)])

            else:
                gt_comp = np.concatenate([np.array(gt), np.array(gt), np.array(gt), np.array(gt), np.array(gt), np.array(gt)])

            #sen p
            pos_index = gt_comp == 1
            obs = np.array([
                [
                    np.sum(preds_comp[pos_index] * preds_comp_ai[pos_index]),
                    np.sum(preds_comp[pos_index] * (1 - preds_comp_ai[pos_index])),
                ],
                [
                    np.sum((1 - preds_comp[pos_index]) * preds_comp_ai[pos_index]),
                    np.sum((1 - preds_comp[pos_index]) * (1 - preds_comp_ai[pos_index]))
                ],
            ])
            result = mcnemar(obs)
            pvalue_sen = result.pvalue

            # spec p
            pos_index = gt_comp == 0
            obs = np.array([
                [
                    np.sum(preds_comp[pos_index] * preds_comp_ai[pos_index]),
                    np.sum(preds_comp[pos_index] * (1 - preds_comp_ai[pos_index])),
                ],
                [
                    np.sum((1 - preds_comp[pos_index]) * preds_comp_ai[pos_index]),
                    np.sum((1 - preds_comp[pos_index]) * (1 - preds_comp_ai[pos_index]))
                ],
            ])
            result = mcnemar(obs)
            pvalue_spec = result.pvalue

            # acc p
            obs = np.array([
                [
                    np.sum((preds_comp==gt_comp) * (preds_comp_ai==gt_comp)),
                    np.sum((preds_comp==gt_comp) * (1 - (preds_comp_ai==gt_comp))),
                ],
                [
                    np.sum((1 - (preds_comp==gt_comp)) * (preds_comp_ai==gt_comp)),
                    np.sum((1 - (preds_comp==gt_comp)) * (1 - (preds_comp_ai==gt_comp)))
                ],
            ])
            result = mcnemar(obs)
            pvalue_acc = result.pvalue

            p_result.append([task, head, pvalue_acc, pvalue_sen, pvalue_spec])

        header = ['task', 'doc', 'accuracy_p','recall_p', 'spec_p']

    if compare_ai:
        output_name = 'compare_ai_doc_p.csv'
    else:
        output_name = 'compare_doc_p.csv'
    pd.DataFrame(p_result).to_csv(os.path.join(SAVE_PATH, output_name), header=header)

