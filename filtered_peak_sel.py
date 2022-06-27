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

def zero_sum (input_list):
    result = 0
    for val in input_list:
        result += val
    
    if result == 0:
        return result

    else:
        return 1

def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

for k in range(len(DB_PATH)):
    R_PATH = DEFAULT_PATH + DB_PATH[k] + "/"

    exclude_record = ["bw", "em", "ma"]

    dict_ann = []
    windowed_list = []
    record_ann = []
    longest = 0.

    record_mat = io.loadmat(DEFAULT_PATH)

    record_mat_keys = list(record_mat.keys())
    record_mat_keys = record_mat_keys[3:]
    
    for j in tqdm(range(len(record_mat_keys))):
        zero_padded_list = []
        dict_ann = []
        temp_rpath = R_PATH + record_mat_keys[j]

        temp_annpath = ANN_PATH + str(record_mat_keys[j][-3:])
        temp_pickle = MAT_PATH + DB_PATH[k] + record_mat_keys[j] + ".pkl"

        # Read original sample by rdsamp function
        record_sg = record_mat[record_mat_keys[j]]

        # Got R-R Peak by rdann funciton
        record_ann = list(wfdb.rdann(temp_annpath, 'atr', sampfrom=0).sample)[1:]
        record_ann_sym = list(wfdb.rdann(temp_annpath, 'atr', sampfrom=0).symbol)[1:]

        interval = wp.ann2rr(temp_annpath, 'atr', as_array=True)
        
        for i in range(len(record_ann)):            
            try:
                pre_add = record_ann[i - 1]
                post_add = record_ann[i + 1]
            except IndexError:
                pre_add = record_ann[i - 1]
                post_add = record_ann[-1]

            avg_div = (interval[i - 1] + interval[i]) / 2 
            cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2)
            cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2)
            
            check_ann = record_ann_sym[i]
            if check_ann in NORAML_ANN:
                record_ann_sym[i] = "N"
            elif check_ann in SUPRA_ANN:
                record_ann_sym[i] = "S"
            elif check_ann in VENTRI_ANN:
                record_ann_sym[i] = "V"
            elif check_ann in FUSION_ANN:
                record_ann_sym[i] = "F"
            elif check_ann in UNCLASS_ANN:
                record_ann_sym[i] = "Q"
            else:
                continue
            
            windowed_list = flatter(record_sg[cut_pre_add:cut_post_add])
            if zero_sum(windowed_list) == 0:
                continue

            cut_it_off = int((428 - len(windowed_list)) / 2)

            if len(windowed_list) > 428: 
                cut_it_off = 0
                cut_pre_add = record_ann[i] - int(428 / 2)
                cut_post_add = record_ann[i] + int(428 / 2) 
                windowed_list = flatter(record_sg[cut_pre_add:cut_post_add])
                zero_padded_list.append(windowed_list)
                
            else:
                cut_it_off = int((428 - len(windowed_list)) / 2)

                if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
                    zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
                else:
                    zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
            
            # plt.plot(zero_padded_list[-1])
            # plt.show()
            dict_ann.append(record_ann_sym[i])

        ann_dict = {
            0 : zero_padded_list,
            1 : dict_ann
        }

        with open(temp_pickle, "wb") as f:
            pickle.dump(ann_dict, f)
        # print(temp_pickle, " SAVED!")

# Checking code
DATA_PATH = "./pickle_mat/"

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in tqdm(range(len(record_list))):
    temp_path = DATA_PATH + "mitPssigP_" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)
        for i in range(len(pickle_input[0])):
            X.append(pickle_input[0][i])

        for i in range(len(pickle_input[1])):
            check_ann = pickle_input[1][i]
            temp_ann_list = list()
            if check_ann == "N":            # Normal
                temp_ann_list.append(0)

            elif check_ann == "S":          # Supra-ventricular
                temp_ann_list.append(1)

            elif check_ann == "V":          # Ventricular
                temp_ann_list.append(2)

            elif check_ann == "F":          # False alarm
                temp_ann_list.append(3)

            elif check_ann == "Q":          # Unclassed 
                temp_ann_list.append(4)
            
            else:
                temp_ann_list.append(9)
            y.append(temp_ann_list)

X = np.array(X)
y = np.array(y)

print(np.unique(y))
print(X.shape)
print(y.shape)