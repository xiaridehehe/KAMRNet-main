'''
This script is to calculate the roc

'''


import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix


import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpkg
from rpy2.robjects.vectors import StrVector
import math
import os
import csv
#
from multiprocessing.pool import Pool

from config.options import TASKLIST, thresholds, TaskFullNameMap
from utils.select_by_id import match_id

pROC = rpkg.importr('pROC')
irr = rpkg.importr('irr')

pandas2ri.activate()

from ci.cal_metrics import cal_metrics



import matplotlib
from matplotlib import pyplot as plt
import matplotlib.font_manager as font_manager

plt.rcParams['font.serif'] = ['Times New Roman']



def cal_roc(task, prob_file, label_df):
    a = pd.read_csv(prob_file)
    b = label_df
    a.sort_values(by='ID', axis=0, inplace=True)
    a.reset_index(drop=True, inplace=True)
    b.sort_values(by='ID', axis=0, inplace=True)
    b.reset_index(drop=True, inplace=True)
    # check if a and b have same first column
    # select the columns in TASKLIST
    b = b[['ID',*TASKLIST]]
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
    roc = pROC.roc(gt, pred, ci=True)

    sen_list = [round(x, 3) for x in roc[1].tolist()]
    sen_list.reverse()

    # choose by the threshold
    spec_best = pROC.coords(roc, x=thresholds[task], input='threshold', ret='specificity', transpose=False)[0]
    sens_best = pROC.coords(roc, x=thresholds[task], input='threshold', ret='sensitivity', transpose=False)[0]
    print(task, roc.rx2('auc')[0])

    acc_best = pROC.coords(roc, x=spec_best, input='specificity', ret='accuracy', transpose=False)[0]
    acc_ci_array = pROC.ci(roc, of='coords', x=spec_best, input='specificity', ret='accuracy', transpose=False)[0]
    acc_low_best, acc_med_best, acc_high_best = acc_ci_array[0][0], acc_ci_array[0][1], acc_ci_array[0][2]

    auc_low, auc, auc_high = pROC.ci(roc)
    tpr = roc.rx2('sensitivities')
    fpr = 1-roc.rx2('specificities')


    M = np.concatenate((np.array([pred]), np.array([gt])), axis=0).transpose()
    _, num_subject, raters, irr_name, kappa_score, stats_name, z, p_value = irr.kappa2(M, 'unweighted')
    position_ci = ro.r('positon_ci <- qnorm(c(.025,.975))')
    kappa_low, kappa_high = kappa_score + position_ci * (kappa_score / z)

    stats_dict = {'doc_name': 'AI MODEL',
                  'tpr': tpr,
                  'fpr': fpr, 
                  # 'roc': roc,
                  'auc': auc,
                  'auc_low': auc_low,
                #   'auc_med': auc_med,
                  'auc_high': auc_high,
                  'sen': sens_best,
                  'spe': spec_best,
                #   'sen_low': se_low_best,
                  # 'sen_med': se_med_best,
                #   'sen_high': se_high_best,
                #   'spe_low': sp_low_best,
                  # 'spe_med': sp_med_best,
                #   'spe_high': sp_high_best,
                  'acc': acc_best,
                  'acc_low': acc_low_best,
                  'acc_high': acc_high_best,
                  'kappa': kappa_score[0],
                  'kappa_low': kappa_low,
                  'kappa_high': kappa_high,
                #   'sen_ci': sen_ci,
                  }
    return  stats_dict

def plot_roc(stats_dict1, stats_dict2, title, output_file, ax=None, return_ax=False):

    plt.tight_layout()
    
    fpr1 = stats_dict1['fpr']
    tpr1 = stats_dict1['tpr']
    auc1 = stats_dict1['auc']
    auc1_low = stats_dict1['auc_low']
    auc1_high = stats_dict1['auc_high']
    fpr2 = stats_dict2['fpr']
    tpr2 = stats_dict2['tpr']
    auc2 = stats_dict2['auc']
    auc2_low = stats_dict2['auc_low']
    auc2_high = stats_dict2['auc_high']
        
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(12, 9))
    ax.plot(fpr1, tpr1, 'red', linewidth=2.5, label='AUC = %.3f' % auc1 + ' (95%% CI %.3f-%.3f)' % (auc1_low, auc1_high))
    ax.plot(fpr2, tpr2, 'blue', linewidth=2.5, label='AUC = %.3f' % auc2+ ' (95%% CI %.3f-%.3f)' % (auc2_low, auc2_high))

    # draw a diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2.5)

    label_size = 30
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=18)
    ax.set_xlim([-0.005, 1.005])
    ax.set_ylim([0.0, 1.005])
    ax.set_xlabel('1-Specificity',fontsize=20)
    ax.set_ylabel('Sensitivity',fontsize=20)
    ax.set_title(title, fontsize=24)
    if return_ax:
        return ax
    if output_file.endswith('.svg'):
        plt.savefig(output_file, dpi=1000, format='svg')
    elif output_file.endswith('.png'):
        plt.savefig(output_file, dpi=1000, format='png')
    plt.close()

def write_roc_to_file(stats_dict0, stats_dict1, subgroup_desc, output_file):
    fpr0 = stats_dict0['fpr']
    tpr0 = stats_dict0['tpr']
    auc0 = stats_dict0['auc']
    auc0_low = stats_dict0['auc_low']
    auc0_high = stats_dict0['auc_high']

    fpr1 = stats_dict1['fpr']
    tpr1 = stats_dict1['tpr']
    auc1 = stats_dict1['auc']
    auc1_low = stats_dict1['auc_low']
    auc1_high = stats_dict1['auc_high']

    # utf-8
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(r'1-Specificity, Sensitivity, 1-Specificity, Sensitivity'+'\n')
        f.writelines(', , , \n')
        f.writelines(f', {subgroup_desc[0]} (AUC: {auc0:.3f}; {auc0_low:.3f}-{auc0_high:.3f}), , {subgroup_desc[1]} (AUC: {auc1:.3f}; {auc1_low:.3f}-{auc1_high:.3f})\n')
        buffer = []
        for i in range(len(fpr0)):
            buffer.append(f'{fpr0[i]}, {tpr0[i]}')
        if (len(fpr0)<len(fpr1)):
            for i in range(len(fpr0)):
                buffer[i] += f', {fpr1[i]}, {tpr1[i]}\n'
            for i in range(len(fpr0), len(fpr1)):
                buffer.append(f', , {fpr1[i]}, {tpr1[i]}\n')
        else:
            for i in range(len(fpr1)):
                buffer[i] += f', {fpr1[i]}, {tpr1[i]}\n'
            for i in range(len(fpr1), len(fpr0)):
                buffer[i] += '\n'
        f.writelines(buffer)



def run(output_dir, prob_file, label_file, subgroup_type, plot=False):
    
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, subgroup_type)
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)



    writer = pd.ExcelWriter(os.path.join(output_dir, f'roc_{subgroup_type}.xlsx'))


    for i, task in enumerate(TASKLIST):#
        
        dict_list = []
        if subgroup_type  == 'all':
            label_df = pd.read_csv(label_file)
            result_dict = cal_roc(task, prob_file, label_df)
            subgroup_desc = ['All', 'All']
            dict_list.append(result_dict)
            dict_list.append(result_dict) # duplicate for plot_roc
        else:
            for subgroup_num in [0, 1]:
                subgroup_df = pd.read_csv(label_file)
                assert subgroup_type in subgroup_df.columns, f'{subgroup_type} not in {subgroup_df.columns}'
                if subgroup_type == 'age':
                    subgroup_desc = ['Age ≤ 40 years', 'Age > 40 years']
                    if subgroup_num == 0:
                        subgroup_df = subgroup_df[subgroup_df['age']<=40]
                    else:
                        subgroup_df = subgroup_df[subgroup_df['age']>40]
                elif subgroup_type == 'sex':
                    subgroup_desc = ['Female', 'Male']
                    if subgroup_num == 0:
                        subgroup_df = subgroup_df[subgroup_df['sex']==0] #female
                    else:
                        subgroup_df = subgroup_df[subgroup_df['sex']==1] #male
                else :
                    raise ValueError('subgroup_type should be sex or age')
                
                result_dict = cal_roc(task, prob_file, subgroup_df)
                dict_list.append(result_dict)
        
        if plot:
            output_img = os.path.join(output_dir, f'{task}_roc.png')
            plot_roc(dict_list[0], dict_list[1], task, output_img)

        output_csv = os.path.join(output_dir, f'{task}_{subgroup_type}_roc.csv')
        write_roc_to_file(dict_list[0], dict_list[1], subgroup_desc, output_csv)
        # merge csv to excel
        df = pd.read_csv(output_csv)
        df.to_excel(writer, sheet_name=TaskFullNameMap[task], index=False)
    writer.close()

