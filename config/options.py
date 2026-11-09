

TASKLIST = ['MENI', 'CART', 'ACL', 'PCL', 'MCL', 'LCL', 'IFP', 'PLICA', 'CYST']
TaskToIndx = {task: i for i, task in enumerate(TASKLIST)}
TaskFullNameMap = {'MENI': 'Meniscal tear', 'CART': 'Cartilage defect', 'ACL': 'ACL tear', 'PCL': 'PCL injury', 'MCL': 'MCL injury', 'LCL': 'LCL injury', 'IFP': 'IFP injury', 'PLICA': 'Synovial plica', 'CYST': 'Cyst'}


# thresholds = {'MENI':0.5, 'ACL':0.5, 'CART':0.5, 'PCL':0.5, 'MCL':0.5, 'LCL':0.5, 'EFFU':0.5, 'CONT':0.5, 'PLICA':0.5,'CYST':0.5, 'IFP':0.5, 'PR':0.5, 'all':0.5} #0.5
thresholds = {'MENI':0.8, 'ACL':0.8, 'CART':0.75, 'PCL':0.6, 'MCL':0.75, 'LCL':0.35, 'EFFU':0.93, 'CONT':0.5, 'PLICA':0.55,'CYST':0.7, 'IFP':0.49, 'PR':0.5, 'all':0.5} #Val
