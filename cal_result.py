import os
import pandas as pd
from ci import generate_result, cal_auc_ci_all, cal_auc_ci3, cal_p, mcnemar_test_p_value, cal_roc_curve, draw_conf_matrix, cal_nri

import matplotlib.pyplot as plt
from config.mypathlib import dataset3_path, nanhai_path, internal_path, ExpRoot
from utils.select_by_id import select
from utils import extract_onetable



dataset_name = "nanhai"
dataset_path = nanhai_path
cmap = plt.cm.Oranges

# dataset_name = "internal"
# dataset_path = internal_path
# cmap = plt.cm.Blues

# dataset_name = "3dataset"
# dataset_path = dataset3_path
# cmap = plt.cm.Greens

mode = 'test'

workspace = os.path.join(ExpRoot, dataset_name)
prob_file = os.path.join(workspace, f'mrnet_ROIc2_all_{mode}.csv')
label_file = getattr(dataset_path,f'{mode}_label')
bool_file = os.path.join(workspace, f'mrnet_ROIc2_all_{mode}_bool.csv')
doc_result_folder = os.path.join(workspace, "doc_result")
# doc_result_folder = dataset_path.doc_result_folder
doc_ci_folder = os.path.join(workspace, "doc_ci")
subgroup_file = dataset_path.subgroup_label


# gether result for tasks
# generate_result.gather_result(path = workspace)


# select by label
# select(prob_file, label_file, prob_file)


# generate bool result
# generate_result.run(result_file=prob_file,
#     gt_file = label_file,
#     output_file=bool_file)

# plot confusion matrix
# draw_conf_matrix.run(prob_file=prob_file,
#                      label_file=label_file,
#                      out_dir=os.path.join(workspace, 'confusion_matrix'),
#                      cmap=cmap,
#                      )


# calculate doctor NRI
# cal_nri.run(output_dir=workspace,
#               label_file=label_file,
#                 doc_result_path=doc_result_folder)



# calculate auc sensitivity and specificity for prediction
# cal_auc_ci3.run(output_dir=workspace,
#                 prob_file = prob_file,
#                 label_file= label_file,
#                 )


# extract one table
# extract_onetable.run(sourcefile=os.path.join(doc_result_folder, doc_result_dir = ''),
#                       doc_result_dir=doc_result_folder)


# match the doctor result
# for suf in ['woAI', 'wAI']:
#     inpath = os.path.join(dataset_path.doc_result_folder, suf)
#     outpath = os.path.join(workspace, 'doc_result', suf)
#     if os.path.exists(outpath) is False:
#         os.mkdir(outpath)
#     for file in os.listdir(inpath):
#         if file.endswith("csv"):
#             doc_file = os.path.join(inpath, file)
#             output_file = os.path.join(outpath, file)
#             select(doc_file, label_file, output_file)


# calculate auc sensitivity and specificity for doctor result
# for suf in ['woAI', 'wAI']:
#     inpath = os.path.join(doc_result_folder, suf)
#     outpath = os.path.join(doc_ci_folder, suf)
#     if os.path.exists(outpath) is False:
#         os.mkdir(outpath)
#     for file in os.listdir(inpath):
#         if file.endswith("csv"):
#             doc_file = os.path.join(inpath, file)
#             output_dir = os.path.join(outpath, file.split('.')[0])
#             if os.path.exists(output_dir) is False:
#                 os.mkdir(output_dir)
#             cal_auc_ci3.run(output_dir=output_dir,
#                             prob_file=doc_file,
#                             label_file=label_file,
#                             )


# calculate auc sensitivity and specificity for doctor result
# cal_auc_ci_all.run(output_dir=doc_ci_folder,
#                    label_file = label_file,
#                    doc_result_folder= doc_result_folder)


# # calculate p value of AI and doctor
# mcnemar_test_p_value.run(output_dir=workspace,
#                          label_file=label_file,
#                          bool_file=bool_file,
#                          doc_result_path=doc_result_folder)


# # calculate p value of w/wo AI
# mcnemar_test_p_value.run(output_dir=workspace,
#                          label_file=label_file,
#                          bool_file=None,
#                          doc_result_path=doc_result_folder)

# output roc curve for AI
# roc_output_dir = os.path.join(workspace, 'roc')
# cal_roc_curve.run(output_dir=os.path.join(workspace, 'roc'),
#                   prob_file=prob_file,
#                   label_file=label_file,
#                   subgroup_type='all',
#                   plot=False,
#                 )


# output subgroup roc
# for subgroup_type in ['age', 'sex']:
#     cal_roc_curve.run(output_dir=os.path.join(workspace, 'roc'),
#                       prob_file=prob_file,
#                       subgroup_file = subgroup_file,
#                       subgroup_type = subgroup_type,
#                     )
                      

