# Cutting as a beat that just raw signal.
# That's it.

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import wfdb.processing as wp
from tqdm import tqdm
import numpy as np
import random
import pickle
import wfdb

PICKLE_PATH = "./pickle_rand/"
DEFAULT_PATH = "./data/"
DB_PATH = ['mit']
EXTRA_NP = np.array(0)

NORAML_ANN = ['N', 'L', 'R', 'e', 'j']
SUPRA_ANN = ['A', 'a', 'J', 'S']
VENTRI_ANN = ['V', 'E']
FUSION_ANN = ['F']
UNCLASS_ANN = ['/', 'f', 'Q']

np.random.seed(42)

def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

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

print("[INFO] Starting Cutting beat and make it delicious pickle")
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
        
        if check_ann in NORAML_ANN:
            record_ann_sym[i] = "N"
            sigN.append(flatter(record_sg[pre_add:post_add]))
            annN.append(record_ann_sym[i])

        elif check_ann in SUPRA_ANN:
            record_ann_sym[i] = "S"
            sigS.append(flatter(record_sg[pre_add:post_add]))
            annS.append(record_ann_sym[i])

        elif check_ann in VENTRI_ANN:
            record_ann_sym[i] = "V"
            sigV.append(flatter(record_sg[pre_add:post_add]))
            annV.append(record_ann_sym[i])

        elif check_ann in FUSION_ANN:
            record_ann_sym[i] = "F"
            sigF.append(flatter(record_sg[pre_add:post_add]))
            annF.append(record_ann_sym[i])

        elif check_ann in UNCLASS_ANN:
            record_ann_sym[i] = "Q"
            sigQ.append(flatter(record_sg[pre_add:post_add]))
            annQ.append(record_ann_sym[i])

        else:
            continue
    
    dict_ann.append(record_ann_sym[i])

sigN_np = np.array(sigN)
annN_np = np.array(annN)

sigS_np = np.array(sigS)
annS_np = np.array(annS)

sigV_np = np.array(sigV)
annV_np = np.array(annV)

sigF_np = np.array(sigF)
annF_np = np.array(annF)

sigQ_np = np.array(sigQ)
annQ_np = np.array(sigQ)

print("[SIZE]\t\tsigN : {}\t\tannN : {}".format(sigN_np.shape, annN_np.shape))

sigN_ran = random.sample(sigN, 7000)
annN_ran = random.sample(annN, 7000)
sigN_np = np.array(sigN_ran)
annN_np = np.array(annN_ran)

sigV_ran = random.sample(sigV, 7000)
annV_ran = random.sample(annV, 7000)
sigV_np = np.array(sigV_ran)
annV_np = np.array(annV_ran)

sigQ_ran = random.sample(sigQ, 7000)
annQ_ran = random.sample(annQ, 7000)
sigQ_np = np.array(sigQ_ran)
annQ_np = np.array(annQ_ran)

Xtr_N, Xte_N, Ytr_N, Yte_N = train_test_split(sigN_np, annN_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_S, Xte_S, Ytr_S, Yte_S = train_test_split(sigS_np, annS_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_V, Xte_V, Ytr_V, Yte_V = train_test_split(sigV_np, annV_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_F, Xte_F, Ytr_F, Yte_F = train_test_split(sigF_np, annF_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_Q, Xte_Q, Ytr_Q, Yte_Q = train_test_split(sigQ_np, annQ_np, test_size=0.3, random_state=42, shuffle=True)

print("- "*35)

print("[SIZE]\t\tXtr_N : {}\t\tXte_N : {}\n\t\tYtr_N : {}\t\t\tYte_N : {}".format(Xtr_N.shape, Xte_N.shape, Ytr_N.shape, Yte_N.shape))
print("- "*35)

print("[SIZE]\t\tXtr_S : {}\t\tXte_S : {}\n\t\tYtr_S : {}\t\t\tYte_S : {}".format(Xtr_S.shape, Xte_S.shape, Ytr_S.shape, Yte_S.shape))
print("- "*35)

print("[SIZE]\t\tXtr_V : {}\t\tXte_V : {}\n\t\tYtr_V : {}\t\t\tYte_V : {}".format(Xtr_V.shape, Xte_V.shape, Ytr_V.shape, Yte_V.shape))
print("- "*35)

print("[SIZE]\t\tXtr_F : {}\t\tXte_F : {}\n\t\tYtr_F : {}\t\t\tYte_F : {}".format(Xtr_F.shape, Xte_F.shape, Ytr_F.shape, Yte_F.shape))
print("- "*35)

print("[SIZE]\t\tXtr_Q : {}\t\tXte_Q : {}\n\t\tYtr_Q : {}\t\t\tYte_Q : {}".format(Xtr_Q.shape, Xte_Q.shape, Ytr_Q.shape, Yte_Q.shape))
print("- "*35)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Random zero-padding part
R_PATH = DEFAULT_PATH + DB_PATH[0] + "/"

dict_ann = []
windowed_list = []
zero_padded_list = []
for j in tqdm(range(len(Xtr_N))):
    temp_rpath = R_PATH + str(j)
    temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

    windowed_list = Xtr_N[j]
    cut_it_off = int((428 - len(windowed_list)) / 2)

    if np.sum(windowed_list) == 0:
        continue

    if len(windowed_list) > 428: 
        cut_it_off = 0            
        
    else:
        cut_it_off = int((428 - len(windowed_list)) / 2)

        random_pre_add = np.random.randint(0, 429 - len(windowed_list))
        random_post_add = 428 - (random_pre_add + len(windowed_list))
    
        if random_post_add > 428:
            random_post_add = 427 - random_post_add

        windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
    windowed_list = np.array(windowed_list, dtype=np.float64)
    zero_padded_list.append(windowed_list[:428])
    
    plt.plot(zero_padded_list[-1])
    plt.show()
    
Xtr_N = np.array(zero_padded_list)

dict_ann = []
windowed_list = []
zero_padded_list = []
for j in tqdm(range(len(Xte_N))):
    # Got R-R Peak by rdann funciton
    windowed_list = Xte_N[j]
    cut_it_off = int((428 - len(windowed_list)) / 2)

    if len(windowed_list) > 428: 
        cut_it_off = 0
        zero_padded_list.append(windowed_list)
        
    else:
        cut_it_off = int((428 - len(windowed_list)) / 2)

        if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
            zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
        else:
            zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
        
    plt.plot(zero_padded_list[-1])
    plt.show()
        

print("- "*15 + "Random Zero-padding applied" + " -" * 15)
print("[SIZE]\t\tXtr_N : {}\t\t\tXte_N : {}\n\t\tYtr_N : {}\t\t\tYte_N : {}".format(Xtr_N.shape, Xte_N.shape, Ytr_N.shape, Yte_N.shape))
print("- "*55)
