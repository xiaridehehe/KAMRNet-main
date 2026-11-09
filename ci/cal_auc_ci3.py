'''
This script is to calculate the confidence interval of ACC, sensitivity, specificity

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
#
from multiprocessing.pool import Pool

from config.options import TASKLIST, thresholds

pROC = rpkg.importr('pROC')
irr = rpkg.importr('irr')

pandas2ri.activate()

from ci.cal_metrics import cal_metrics


def process_one(inputs):
    sen, spe, roc = inputs
    se_low, se_med, se_high = pROC.ci(roc, of='se', specificities=spe)[0]
    spe_low, spe_med, spe_high = pROC.ci(roc, of='sp', sensitivities=sen)[0]

    return spe, se_low, se_med, se_high, spe_low, spe_med, spe_high





def run(output_dir, prob_file, label_file, output_prefix = ''):
    ff_csv = open(os.path.join(output_dir, f'{output_prefix}test_sen_spe_CI.csv'), 'w+')
    ff_csv.write('task, doc_name, auc(low-high), acc(low-hight),sen(low-hight),spe(low-hight),kappa(low-hight) \n')
    for task in TASKLIST:#


        result_dict = [cal_metrics(task, prob_file, label_file)]
        for result in result_dict:
            ff_csv.write(f'{task},{result['doc_name']},')
            ff_csv.write('%.3f(%.3f-%.3f), %.3f(%.3f-%.3f), %.3f(%.3f-%.3f), %.3f(%.3f-%.3f),' % (# %.4f(%.4f-%.4f)
                result['auc']*100, result['auc_low']*100, result['auc_high']*100,
                result['acc']*100, result['acc_low']*100, result['acc_high']*100,
                result['sen']*100, result['sen_low']*100, result['sen_high']*100,
                result['spe']*100, result['spe_low']*100, result['spe_high']*100,
                result['kappa']*100, result['kappa_low']*100, result['kappa_high']*100,
            ))
            ff_csv.write('\n')

    ff_csv.close()
