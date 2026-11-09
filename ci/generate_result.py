'''
generate binary result from probability result
'''


import os
import pandas as pd
import numpy as np
from sklearn import metrics

from config.options import TASKLIST as tasklist, thresholds

def gather_result(path, outputname):
    files = os.listdir(path)
    task_df_dict = {}
    # write first row as [id, task1, task2, ...]
    result = pd.DataFrame(columns=['ID'] + tasklist)
    for file in files:
        if file.endswith('.csv'):
            task = file.split('_')[2]
            if task not in tasklist:
                continue
            file_path = os.path.join(path, file)
            df = pd.read_csv(file_path)
            task_df_dict[task] = df
    # iterate over all the dataframes and merge them, match the first column
    Buffer = []
    for i in range(len(task_df_dict['MENI'])):
        id = task_df_dict['MENI'].iloc[i, 0]
        row = [id]
        for task in tasklist:
            # find the line with the same id
            line = task_df_dict[task][task_df_dict[task].iloc[:, 0] == id]
            if len(line) == 0:
                print('id not found:', id)
                return
            # get the label
            label = line.iloc[0, 1]
            row.append(label)
        Buffer.append(row)
    result = pd.DataFrame(Buffer, columns=['ID'] + tasklist)
    # sort by ID
    result.sort_values(by='ID', axis=0, inplace=True)
    result.to_csv(os.path.join(path, outputname), index=False)

def run(result_file, gt_file, output_file):

    test_file = pd.read_csv(result_file)
    gt_file = pd.read_csv(gt_file)



    test_file.sort_values(by='ID', axis=0, inplace=True)
    test_file.reset_index(drop=True, inplace=True)
    gt_file.sort_values(by='ID', axis=0, inplace=True)
    gt_file.reset_index(drop=True, inplace=True)


    inference_result = test_file.copy()



    for task in tasklist:
        th = thresholds[task]
        task_result = test_file[task]
        inference_result[task] = (task_result > th).astype(int)
    # re-order the columns by tasklist:
    inference_result = inference_result[['ID'] + tasklist]
    inference_result.sort_values(by='ID', axis=0, inplace=True)
    inference_result.to_csv(output_file, index=False)



