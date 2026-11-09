import os


ExpRoot = f""

model_weight_path = ''

trained_models = {
    'MENI': "",
    'ACL': "",
    'CART': "",
    'PCL': "",
    'MCL': "",
    'LCL': "",
    'EFFU': "",
    'CONT': "",
    'PLICA':"",
    'CYST': "",
    'IFP': "",
    'PR': "",
}


# datapath

data_root = ''

cache_root = os.path.join(data_root, 'cls_cache')

# Build your own data path here


class Internal_Path():
    def __init__(self):
        self.data = ""
        self.train_label = ""
        self.val_label = ""
        self.test_label = ""
        self.subgroup_label = ""
        self.roi = ""
        self.cache = ""
internal_path = Internal_Path()

class Nanhai_Path():
    def __init__(self):
        root = ""
        self.data = ""
        self.train_label = ""
        self.val_label = ""
        self.test_label = ""
        self.roi = ""
        self.doc_result_folder = ""
        self.subgroup_label = ""
        self.cache = ""
nanhai_path = Nanhai_Path()

class Data3_Path():
    def __init__(self):
        root = ""
        self.data = root
        self.train_label = ""
        self.val_label = ""
        self.test_label = ""
        self.roi = ""
        self.doc_result_folder = ""
        self.subgroup_label = ""
        self.cache = ""
dataset3_path = Data3_Path()