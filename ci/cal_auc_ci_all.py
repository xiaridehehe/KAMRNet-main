'''
calculate doctor's auc ci

'''


import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import time
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
from ci.cal_metrics import cal_metrics

pROC = rpkg.importr('pROC')
irr = rpkg.importr('irr')

pandas2ri.activate()



def process_one(inputs):
    sen, spe, roc = inputs
    se_low, se_med, se_high = pROC.ci(roc, of='se', specificities=spe)[0]
    spe_low, spe_med, spe_high = pROC.ci(roc, of='sp', sensitivities=sen)[0]

    return spe, se_low, se_med, se_high, spe_low, spe_med, spe_high


def metrics(task, labelfile, doc_result_folder):
    doc_result_dict = []
    data_path = doc_result_folder
    junior_AI = []
    junior_AI= []
    senior = []
    senior_AI = []
    for path_dir in ['woAI', 'wAI']:
        for doc_result in os.listdir(os.path.join(data_path,path_dir)):
            print(doc_result)
            doc_name = doc_result.split('.')[0]
            doc_result_file = os.path.join(data_path, path_dir, doc_result)
            stats_dict = cal_metrics(task=task,
                                     prob_file=doc_result_file,
                                     label_file=labelfile)
            stats_dict['doc_name'] = doc_name
            doc_result_dict.append(stats_dict)
            


    stats_dict_junior_ai = {'doc_name': 'Junior_AI','sen': 0,'spe': 0,'sen_low': 0,'sen_high': 0, 'spe_low': 0,
                  'spe_high': 0,'acc': 0, 'acc_low': 0, 'acc_high': 0,'kappa': 0,'kappa_low': 0,'kappa_high': 0}
    stats_dict_senior_ai = {'doc_name': 'Senior_AI','sen': 0,'spe': 0,'sen_low': 0,'sen_high': 0, 'spe_low': 0,
                  'spe_high': 0,'acc': 0, 'acc_low': 0, 'acc_high': 0,'kappa': 0,'kappa_low': 0,'kappa_high': 0}
    stats_dict_junior = {'doc_name': 'Junior','sen': 0,'spe': 0,'sen_low': 0,'sen_high': 0, 'spe_low': 0,
                  'spe_high': 0,'acc': 0, 'acc_low': 0, 'acc_high': 0,'kappa': 0,'kappa_low': 0,'kappa_high': 0}
    stats_dict_senior = {'doc_name': 'Senior','sen': 0,'spe': 0,'sen_low': 0,'sen_high': 0, 'spe_low': 0,
                  'spe_high': 0,'acc': 0, 'acc_low': 0, 'acc_high': 0,'kappa': 0,'kappa_low': 0,'kappa_high': 0}
    for doc_result_item in doc_result_dict:
        if 'QY' in doc_result_item['doc_name'] or 'YHZ' in doc_result_item['doc_name']:
            if 'AI' not in doc_result_item['doc_name']:
                stats_dict_senior['sen'] += doc_result_item['sen']/2
                stats_dict_senior['spe'] += doc_result_item['spe']/2
                stats_dict_senior['sen_low'] += doc_result_item['sen_low'] / 2
                stats_dict_senior['sen_high'] += doc_result_item['sen_high'] / 2
                stats_dict_senior['spe_low'] += doc_result_item['spe_low'] / 2
                stats_dict_senior['spe_high'] += doc_result_item['spe_high'] / 2
                stats_dict_senior['acc'] += doc_result_item['acc'] / 2
                stats_dict_senior['acc_low'] += doc_result_item['acc_low'] / 2
                stats_dict_senior['acc_high'] += doc_result_item['acc_high'] / 2
                stats_dict_senior['kappa'] += doc_result_item['kappa'] / 2
                stats_dict_senior['kappa_low'] += doc_result_item['kappa_low'] / 2
                stats_dict_senior['kappa_high'] += doc_result_item['kappa_high'] / 2
            else:
                stats_dict_senior_ai['sen'] += doc_result_item['sen']/2
                stats_dict_senior_ai['spe'] += doc_result_item['spe']/2
                stats_dict_senior_ai['sen_low'] += doc_result_item['sen_low'] / 2
                stats_dict_senior_ai['sen_high'] += doc_result_item['sen_high'] / 2
                stats_dict_senior_ai['spe_low'] += doc_result_item['spe_low'] / 2
                stats_dict_senior_ai['spe_high'] += doc_result_item['spe_high'] / 2
                stats_dict_senior_ai['acc'] += doc_result_item['acc'] / 2
                stats_dict_senior_ai['acc_low'] += doc_result_item['acc_low'] / 2
                stats_dict_senior_ai['acc_high'] += doc_result_item['acc_high'] / 2
                stats_dict_senior_ai['kappa'] += doc_result_item['kappa'] / 2
                stats_dict_senior_ai['kappa_low'] += doc_result_item['kappa_low'] / 2
                stats_dict_senior_ai['kappa_high'] += doc_result_item['kappa_high'] / 2
        else:
            if 'AI' not in doc_result_item['doc_name']:
                stats_dict_junior['sen'] += doc_result_item['sen'] / 4
                stats_dict_junior['spe'] += doc_result_item['spe'] / 4
                stats_dict_junior['sen_low'] += doc_result_item['sen_low'] / 4
                stats_dict_junior['sen_high'] += doc_result_item['sen_high'] / 4
                stats_dict_junior['spe_low'] += doc_result_item['spe_low'] / 4
                stats_dict_junior['spe_high'] += doc_result_item['spe_high'] / 4
                stats_dict_junior['acc'] += doc_result_item['acc'] / 4
                stats_dict_junior['acc_low'] += doc_result_item['acc_low'] / 4
                stats_dict_junior['acc_high'] += doc_result_item['acc_high'] / 4
                stats_dict_junior['kappa'] += doc_result_item['kappa'] / 4
                stats_dict_junior['kappa_low'] += doc_result_item['kappa_low'] / 4
                stats_dict_junior['kappa_high'] += doc_result_item['kappa_high'] / 4
            else:
                stats_dict_junior_ai['sen'] += doc_result_item['sen'] / 4
                stats_dict_junior_ai['spe'] += doc_result_item['spe'] / 4
                stats_dict_junior_ai['sen_low'] += doc_result_item['sen_low'] / 4
                stats_dict_junior_ai['sen_high'] += doc_result_item['sen_high'] / 4
                stats_dict_junior_ai['spe_low'] += doc_result_item['spe_low'] / 4
                stats_dict_junior_ai['spe_high'] += doc_result_item['spe_high'] / 4
                stats_dict_junior_ai['acc'] += doc_result_item['acc'] / 4
                stats_dict_junior_ai['acc_low'] += doc_result_item['acc_low'] / 4
                stats_dict_junior_ai['acc_high'] += doc_result_item['acc_high'] / 4
                stats_dict_junior_ai['kappa'] += doc_result_item['kappa'] / 4
                stats_dict_junior_ai['kappa_low'] += doc_result_item['kappa_low'] / 4
                stats_dict_junior_ai['kappa_high'] += doc_result_item['kappa_high'] / 4

    doc_result_dict.append(stats_dict_junior)
    doc_result_dict.append(stats_dict_senior)
    doc_result_dict.append(stats_dict_junior_ai)
    doc_result_dict.append(stats_dict_senior_ai)

    return doc_result_dict #stats_dict



def run(output_dir, label_file, doc_result_folder):
    ff_csv = open(os.path.join(output_dir, f'all_doc_test_sen_spe_CI.csv'), 'w+')
    ff_csv.write(
    'task,doc_name,acc(low-hight),acc,sen(low-hight),sen,spe(low-hight),spe,kappa(low-hight) \n')
    for task in [*TASKLIST, 'all']:#


        result_dict = metrics(task, labelfile=label_file, doc_result_folder=doc_result_folder)
        for result in result_dict:
            ff_csv.write(f'{task},{result['doc_name']},')
            ff_csv.write('%.1f(%.1f-%.1f),%.1f, %.1f(%.1f-%.1f),%.1f, %.1f(%.1f-%.1f),%.1f, %.1f(%.1f-%.1f),' % (
                result['acc']*100, result['acc_low']*100, result['acc_high']*100,
                result['acc']*100,
                result['sen']*100, result['sen_low']*100, result['sen_high']*100,
                result['sen']*100,
                result['spe']*100, result['spe_low']*100, result['spe_high']*100,
                result['spe']*100,
                result['kappa']*100, result['kappa_low']*100, result['kappa_high']*100,
            ))
            ff_csv.write('\n')

    ff_csv.close()
