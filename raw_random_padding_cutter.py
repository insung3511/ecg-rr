# Cutting as a beat that just raw signal.
# That's it.

from operator import concat
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import wfdb.processing as wp
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import wfdb

PICKLE_PATH = "./pickle_mat/"
DEFAULT_PATH = "./data/"
DB_PATH = ['mit']
EXTRA_NP = np.array(0)

NORAML_ANN = ['N', 'L', 'R', 'e', 'j']
SUPRA_ANN = ['A', 'a', 'J', 'S']
VENTRI_ANN = ['V', 'E']
FUSION_ANN = ['F']
UNCLASS_ANN = ['/', 'f', 'Q']

np.random.seed(42)

def zero_sum(x_array, y_array):
    x_return_array, y_return_array = [], []
    for i in range(len(x_array)):
        if np.sum(x_array[i]) == 0:
            continue
        else:
            x_return_array.append(x_array[i])
            y_return_array.append(y_array[i])
    return x_return_array, y_return_array

def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

def TrainSetPadding(Xarray, Yarray):
    Xreturn, yreturn = [], []
    for i in range(len(Xarray)):
        beat_list = Xarray[i]
        beat_anno = Yarray[i]

        if np.sum(beat_list) == 0:
            continue

        if len(beat_list) >= 428:
            yreturn.append(beat_anno)
            Xreturn.append(beat_list[:428])
        
        else:
            random_front_add = np.random.randint(0, 428 - len(beat_list))
            random_back_add  = 428 - (random_front_add + len(beat_list))
            yreturn.append(beat_anno) 
            Xreturn.append(
                np.append([0.0], np.pad(beat_list, (random_front_add, random_back_add), 'constant', constant_values=0))[:428]
            )
        
    return np.array(Xreturn), np.array(yreturn)

def TestSetPadding(Xarray, Yarray):
    Xreturn, yreturn = [], []
    for i in range(len(Xarray)):
        beat_list = Xarray[i]
        beat_anno = Yarray[i]

        if np.sum(beat_list) == 0:
            continue

        if len(beat_list) > 428:
            yreturn.append(beat_anno)
            Xreturn.append(beat_list[:428])

        else:
            yreturn.append(beat_anno)
            cutting_off = int((428 - len(beat_list)) / 2)

            if len(np.pad(beat_list, cutting_off, 'constant', constant_values=0)) == 427:
                Xreturn.append(np.append([0.0], np.pad(beat_list, cutting_off, 'constant', constant_values=0)))

            else:
                Xreturn.append(np.pad(beat_list, cutting_off, 'constant', constant_values=0)[:428])
    return np.array(Xreturn), np.array(yreturn)    

def concater(noraml, supra, ventri, fusion, q):
    concate_list = []
    concate_list.append(noraml)
    concate_list.append(supra)
    concate_list.append(ventri)
    concate_list.append(fusion)
    concate_list.append(q)
    
    concate_list = np.array(random.shuffle(flatter(concate_list)))
    print("= = = = = = = = %" + str(concate_list.shape))
    return concate_list

# - - - - - - - - - - - - - - - - - - - - - - - - - 
# Read only beats
R_PATH = DEFAULT_PATH + DB_PATH[0] + "/"

dict_ann = []
windowed_list = []
record_list = []
record_ann = []
longest = 0.

zero_padded_list = []

sigN, sigV, sigS, sigF, sigQ = [],[],[],[],[]
annN, annV, annS, annF, annQ = [],[],[],[],[]

# Read RECORDS txt file
print("[INFO] Read records file from ", R_PATH)
with open(R_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

# Read Records
print("[INFO] Read RECORDS that what it read it")
for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

print("[INFO] Starting Cutting beat and make it shuffle as casino")
for j in tqdm(range(len(record_list))):
    temp_rpath = R_PATH + record_list[j]
    record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[0], sampfrom=0)
    
    # Got R-R Peak by rdann funciton
    record_ann = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).sample)[1:]
    record_ann_sym = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).symbol)[1:]
    interval = wp.ann2rr(temp_rpath, 'atr', as_array=True)
    
    for i in range(len(record_ann)):            
        try:
            pre_add = record_ann[i - 1]
            post_add = record_ann[i + 1]
        except IndexError:
            pre_add = record_ann[i - 1]
            post_add = record_ann[-1]

        check_ann = record_ann_sym[i]
        avg_div = (interval[i - 1] + interval[i]) / 2 
        
        cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2)
        cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2) 
        
        cutted_beat = flatter(record_sg[cut_pre_add:cut_post_add])

        if check_ann in NORAML_ANN:         # Normal
            record_ann_sym[i] = "N"
            sigN.append(cutted_beat)
            annN.append(record_ann_sym[i])

        elif check_ann in SUPRA_ANN:        # Supra-Ventricular
            record_ann_sym[i] = "S"
            sigS.append(cutted_beat)
            annS.append(record_ann_sym[i])

        elif check_ann in VENTRI_ANN:       # Ventricular
            record_ann_sym[i] = "V"
            sigV.append(cutted_beat)
            annV.append(record_ann_sym[i])

        elif check_ann in FUSION_ANN:       # Fusion
            record_ann_sym[i] = "F"
            sigF.append(cutted_beat)
            annF.append(record_ann_sym[i])

        elif check_ann in UNCLASS_ANN:      # Unknown
            record_ann_sym[i] = "Q"
            sigQ.append(cutted_beat)
            annQ.append(record_ann_sym[i])

        else:
            continue
        
    dict_ann.append(record_ann_sym[i])

sigN, annN = zero_sum(sigN, annN)
sigS, annS = zero_sum(sigS, annS)
sigV, annV = zero_sum(sigV, annV)
sigF, annF = zero_sum(sigF, annF)
sigQ, annQ = zero_sum(sigQ, annQ)

sigN_np = np.array(sigN)
annN_np = np.array(annN)

sigS_np = np.array(sigS)
annS_np = np.array(annS)

sigV_np = np.array(sigV)
annV_np = np.array(annV)

sigF_np = np.array(sigF)
annF_np = np.array(annF)

sigQ_np = np.array(sigQ)
annQ_np = np.array(annQ)

Xtr_N, Xte_N, Ytr_N, Yte_N = train_test_split(sigN_np, annN_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_S, Xte_S, Ytr_S, Yte_S = train_test_split(sigS_np, annS_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_V, Xte_V, Ytr_V, Yte_V = train_test_split(sigV_np, annV_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_F, Xte_F, Ytr_F, Yte_F = train_test_split(sigF_np, annF_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_Q, Xte_Q, Ytr_Q, Yte_Q = train_test_split(sigQ_np, annQ_np, test_size=0.3, random_state=42, shuffle=True)

random.seed(42)

# Sorted for the get sample
# Without sorting, got a error that didn't sort.
sigN_ran = random.sample(sorted(Xtr_N), 7000)
annN_ran = random.sample(sorted(Ytr_N), 7000)
Xtr_N = np.array(sigN_ran)
Ytr_N = np.array(annN_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigV_ran = []
annV_ran = []
for sig in range(7000):
    sigV_ran.append(random.choice(Xtr_V))
    annV_ran.append(random.choice(Ytr_V))
Xtr_V = np.array(sigV_ran)
Ytr_V = np.array(annV_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigS_ran = []
annS_ran = []
for sig in range(7000):
    sigS_ran.append(random.choice(Xtr_S))
    annS_ran.append(random.choice(Ytr_S))
Xtr_S = np.array(sigS_ran)
Ytr_S = np.array(annS_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigF_ran = []
annF_ran = []
for sig in range(7000):
    sigF_ran.append(random.choice(Xtr_F))
    annF_ran.append(random.choice(Ytr_F))
Xtr_F = np.array(sigF_ran)
Ytr_F = np.array(annF_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigQ_ran = []
annQ_ran = []
for sig in range(7000):
    sigQ_ran.append(random.choice(Xtr_Q))
    annQ_ran.append(random.choice(Ytr_Q))
Xtr_Q = np.array(sigQ_ran)
Ytr_Q = np.array(annQ_ran)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print("- "*35)

Xtr_N, Ytr_N = TrainSetPadding(Xtr_N, Ytr_N)
Xte_N, Yte_N = TestSetPadding(Xte_N, Yte_N)
print("[SIZE]\t\tXtr_N : {}\t\tXte_N : {}\n\t\tYtr_N : {}\t\t\tYte_N : {}".format(Xtr_N.shape, Xte_N.shape, Ytr_N.shape, Yte_N.shape))
print("- "*35)

Xtr_S, Ytr_S = TrainSetPadding(Xtr_S, Ytr_S)
Xte_S, Yte_S = TestSetPadding(Xte_S, Yte_S)
print("[SIZE]\t\tXtr_S : {}\t\tXte_S : {}\n\t\tYtr_S : {}\t\t\tYte_S : {}".format(Xtr_S.shape, Xte_S.shape, Ytr_S.shape, Yte_S.shape))
print("- "*35)

Xtr_V, Ytr_V = TrainSetPadding(Xtr_V, Ytr_V)
Xte_V, Yte_V = TestSetPadding(Xte_V, Yte_V)
print("[SIZE]\t\tXtr_V : {}\t\tXte_V : {}\n\t\tYtr_V : {}\t\t\tYte_V : {}".format(Xtr_V.shape, Xte_V.shape, Ytr_V.shape, Yte_V.shape))
print("- "*35)

Xtr_F, Ytr_F = TrainSetPadding(Xtr_F, Ytr_F)
Xte_F, Yte_F = TestSetPadding(Xte_F, Yte_F)
print("[SIZE]\t\tXtr_F : {}\t\tXte_F : {}\n\t\tYtr_F : {}\t\t\tYte_F : {}".format(Xtr_F.shape, Xte_F.shape, Ytr_F.shape, Yte_F.shape))
print("- "*35)

Xtr_Q, Ytr_Q = TrainSetPadding(Xtr_Q, Ytr_Q)
Xte_Q, Yte_Q = TestSetPadding(Xte_Q, Yte_Q)
print("[SIZE]\t\tXtr_Q : {}\t\tXte_Q : {}\n\t\tYtr_Q : {}\t\t\tYte_Q : {}".format(Xtr_Q.shape, Xte_Q.shape, Ytr_Q.shape, Yte_Q.shape))
print("- "*35 + "\n")

print("- "*8 + "Final Train-Test split result " + "- "*8)
X_train = concater(Xtr_N, Xtr_S, Xtr_V, Xtr_F, Xtr_Q)
y_train = concater(Ytr_N, Ytr_S, Ytr_V, Ytr_F, Ytr_Q)
X_test = concater(Xte_N, Xte_S, Xte_V, Xte_F, Xte_Q)
y_test = concater(Yte_N, Yte_S, Yte_V, Yte_F, Yte_Q)
print("[SIZE]\t\tX_train : {}\t\tX_test : {}\n\t\ty_train : {}\t\t\ty_test : {}".format(X_train.shape, X_test.shape, y_train.shape, y_test.shape))
print("- "*35)
