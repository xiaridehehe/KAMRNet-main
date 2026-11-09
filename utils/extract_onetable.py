import os
import pandas as pd
from config.options import TASKLIST as tasklist

def run(sourcefile, doc_result_dir):

    df = pd.read_csv(sourcefile)
    wAI_dir = os.path.join(doc_result_dir, 'wAI')
    woAI_dir = os.path.join(doc_result_dir, 'woAI')

    os.makedirs(wAI_dir, exist_ok=True)
    os.makedirs(woAI_dir, exist_ok=True)


    doc_dict = {}
    for col in df.columns[1:]:
        taskname, docname = col.split('_')
        if docname in ['LABEL', 'MODEL']:
            continue
        else:
            if docname not in doc_dict:
                doc_dict[docname] = df["ID"]
            newcol = df[col].rename(taskname)
            doc_dict[docname] = pd.concat([doc_dict[docname], newcol], axis=1)

    for docname in doc_dict:
        doc_dict[docname] = doc_dict[docname][['ID', *tasklist]]
        if '+' in docname:
            doc_dict[docname].to_csv(os.path.join(wAI_dir, docname + '.csv'))
        else:
            doc_dict[docname].to_csv(os.path.join(woAI_dir, docname + '.csv'))
            
        