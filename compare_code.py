import matplotlib.pyplot as plt
import wfdb.processing as wp
import scipy.io as io
from tqdm import tqdm
import numpy as np
import pickle
import wfdb

ANN_PATH = "./data/mit/"
MAT_PATH = "./pickle_mat/"
DEFAULT_PATH = "./data/filtered/mitPs.mat"
DB_PATH = ['mitPs']
EXTRA_NP = np.array(0)

NORAML_ANN = ['N', 'L', 'R', 'e', 'j']
SUPRA_ANN = ['A', 'a', 'J', 'S']
VENTRI_ANN = ['V', 'E']
FUSION_ANN = ['F']
UNCLASS_ANN = ['/', 'f', 'Q']

result_mat = []
result_og =  []
mat_s = str()
mat_w = str()

def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

for k in range(len(DB_PATH)):
    R_PATH = DEFAULT_PATH + DB_PATH[k] + "/"
    
    exclude_record = ["bw", "em", "ma"]

    dict_ann = []
    windowed_list = []
    record_list = []
    record_ann = []
    longest = 0.

    record_mat = io.loadmat(DEFAULT_PATH)

    record_mat_keys = list(record_mat.keys())
    record_mat_keys = record_mat_keys[3:]
    
    for i in range(len(record_mat_keys)):
        mat_s = record_mat_keys[i]
        result_mat = (record_mat[record_mat_keys[i]])

    # Read RECORDS txt file
    print("[INFO] Read records file from ./data/mit/")
    with open('./data/mit/RECORDS') as f:
        record_lines = f.readlines()

    # Read Records
    for i in range(len(record_lines)):
        if record_lines[i].strip() in exclude_record:
            continue
        record_list.append(str(record_lines[i].strip()))


    for j in range(len(record_list)):
        mat_w = record_list[j]
        zero_padded_list = []
        dict_ann = []
        temp_rpath = './data/mit/' + record_list[j]

        # Read original sample by rdsamp function
        record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[0], sampfrom=0)
        result_og = (record_sg)    

print(mat_s)
print(mat_w)
print(np.array(result_mat).shape)
print(np.array(result_og).shape)