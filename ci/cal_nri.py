'''
calcluate net reclassification index
'''


import numpy as np
import pandas as pd
import scipy.stats
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

from matplotlib import pyplot as plt

import rpy2.robjects as ro
from rpy2.robjects import r, pandas2ri
import rpy2.robjects.packages as rpkg
from rpy2.robjects.vectors import StrVector
import math
import os
import io
import re
import contextlib
#
from multiprocessing.pool import Pool

from config.options import TASKLIST, thresholds

pROC = rpkg.importr('pROC')
pdable = rpkg.importr('PredictABEL')

pandas2ri.activate()

from utils.select_by_id import match_id


def run(output_dir, label_file, doc_result_path):
    '''
    if boolfile is provided, compare doctor with AI.
    Otherwise, compare doctor, using their result with and without AI.
    '''
    SAVE_PATH = output_dir

    b = pd.read_csv(label_file)[['ID']+TASKLIST]
    data_path = doc_result_path
    
    out_text = ''
    result = []
    # count = 0

    for head in [ 'junior', 'senior', 'all',]:

        b.sort_values(by='ID', axis=0, inplace=True)
        b.reset_index(drop=True, inplace=True)
        csvdata_gt = ro.conversion.py2rpy(b)

        for task in TASKLIST:#
            task_idx = TASKLIST.index(task)+2 # r language starts from 1, and first column is ID

            wo_pred_gather = []
            w_pred_gather = []
            for doc_result in os.listdir(os.path.join(data_path, 'woAI')):
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
                c = pd.read_csv(w_AI_file)

                a = match_id(a, b)
                c = match_id(c, b)
                wo_pred = ro.conversion.py2rpy(a[task])
                w_pred = ro.conversion.py2rpy(c[task])
                print('*'*20, task, doc_name, '*'*20)
                out_text += '*'*20 + ' ' + task + ' ' + doc_name + ' ' + '*'*20 + '\n'
                text, res = get_nri(csvdata_gt, task_idx, wo_pred, w_pred)
                out_text += text
                mean, lower, higher, p_value = res
                result.append({'task':task, 'doc_name':doc_name, 
                               'nri': mean, 'lower': lower, 'higher': higher, 'p_value': p_value})
                print('*'*60, '\n'*2)
                out_text += '*'*60 + '\n'*2

                wo_pred_gather.append(wo_pred)
                w_pred_gather.append(w_pred)
                
            gathered_gt = pd.concat([b]*len(wo_pred_gather))
            gathered_csvdata_gt = ro.conversion.py2rpy(gathered_gt)
            wo_pred_gather = np.concatenate(wo_pred_gather, axis=0)
            w_pred_gather = np.concatenate(w_pred_gather, axis=0)
            print('*'*20, task, head, '*'*20)
            out_text += '*'*20 + ' ' + task + ' ' + head + ' ' + '*'*20 + '\n'
            text, res = get_nri(gathered_csvdata_gt, task_idx, wo_pred_gather, w_pred_gather)
            out_text += text
            mean, lower, higher, p_value =  res
            result.append({'task':task, 'doc_name':head,
                           'nri': mean, 'lower': lower, 'higher': higher, 'p_value': p_value})
            print('*'*60, '\n'*2)
            out_text += '*'*60 + '\n'*2
    # save the result
    result_df = pd.DataFrame(result)
    result_df.to_csv(os.path.join(SAVE_PATH, 'nri.csv'), index=False)
    with open(os.path.join(SAVE_PATH, 'nri.txt'), 'w') as f:
        f.write(out_text)



def get_nri(csvdata_gt, task_idx, wo_pred, w_pred, display=True):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = pdable.reclassification(csvdata_gt, task_idx, wo_pred, w_pred, [0, 0.5, 1])
    if display:
        print(output.getvalue())
    nriline = re.search(r'NRI(.*?)\n', output.getvalue()).group(1)
    assert nriline is not None, "NRI line not found in output"
    
    _, mean, lower, higher, p_value = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", nriline)

    return output.getvalue(), (float(mean), float(lower), float(higher), float(p_value))