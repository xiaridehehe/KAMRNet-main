import numpy as np
import pandas as pd
import os
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, cohen_kappa_score, recall_score, precision_score

from matplotlib import pyplot as plt

from config.options import TASKLIST, thresholds, TaskFullNameMap
from utils.select_by_id import match_id



def run( prob_file, label_file, task = None, out_dir=None, cmap = plt.cm.Blues, *arg, **kwargs):
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

    if task is not None:
        tasklist = [task,]
    else:
        tasklist = TASKLIST
    for task in tasklist:
        pred = a[task].to_numpy()
        gt = b[task].to_numpy()


        # calcluate confusion matrix
        binary_pred = np.where(pred > thresholds[task], 1, 0)
        cm = confusion_matrix(gt, binary_pred)
        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            outpath = os.path.join(out_dir, f'{task}_confusion_matrix.png')
        else:
            outpath = None
        _draw_matrix(cm = cm, classes=['Positive', 'Negative'],
                    out_path = outpath,
                    cmap=cmap,
                    title=TaskFullNameMap[task],
                    )



def _draw_matrix(cm, classes,
                out_path,
                cmap = plt.cm.Blues,
                normalize=False,
                title='Confusion matrix',
                base_fontsize = 40,
                ):
    BS = base_fontsize # fontsize
    # font arial
    # plt.rcParams['font.sans-serif'] = ['Arial']
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    # transparent background
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title, fontsize=0.8 * BS, pad=0.25*BS)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=0.5 * BS)
    plt.yticks(tick_marks, classes, fontsize=0.5 * BS, rotation=90, verticalalignment='center')

    fmt = '.2f' if normalize else 'd'
    thresh = cm.min() + (cm.max()-cm.min()) *0.6
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=0.5 * BS,
                 )

    # plt.tight_layout()
    plt.ylabel('True label', fontsize=0.5 * BS)
    plt.xlabel('Predicted label', fontsize=0.5 * BS)
    # plt.figure()
    if out_path is None:
        plt.show()
    else:
        # print(f"save to {out_path}")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close()