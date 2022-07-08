# Cutting as a beat that just raw signal.
# That's it.

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

def makeTrainSet(Xarray, yarray, rpath_param):
    for j in (range(len(Xarray))):
        temp_rpath = rpath_param + str(j)
        
        windowed_list = Xarray[j]
        cut_if_off = int((428 - len(windowed_list)) /2)

        if np.sum(windowed_list) == 0:
            continue

        if len(windowed_list) > 428:
            zero_padded_list.append(windowed_list)[:428]
            
        else:
            cut_it_off = int((428 - len(windowed_list)) / 2)

            random_pre_add = np.random.randint(0, 429 - len(windowed_list))
            random_post_add = 428 - (random_pre_add + len(windowed_list))
        
            if random_post_add > 428:
                random_post_add = 427 - random_post_add

            windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))

        zero_padded_list.append(windowed_list[:428])
        test_yarray.append(yarray)
        
    print("[SIZE]\t\tXarray : {}\t\tYarray : {}"
        .format(np.array(zero_padded_list).shape, np.array(test_yarray).shape))

    print("- "*35)
    return zero_append_list, test_yarray

def makeTestSet(Xarray, yarray, rpath_params):
    for j in (range(len(Xarray))):
        temp_rpath = rpath_param + str(j)
        
        windowed_list = Xtr_F[j]
        cut_if_off = int((428 - len(windowed_list)) /2)

        if np.sum(windowed_list) == 0:
            continue

        if len(windowed_list) > 428:
            zero_padded_list.append(windowed_list)[:428]
            
        else:
            cut_it_off = int((428 - len(windowed_list)) / 2)

            if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
                zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
            else:
                zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
        
        test_yarray.append(yarray)

    print("[SIZE]\t\tXarray : {}\t\tYarray : {}"
        .format(np.array(zero_padded_list).shape, np.array(test_yarray).shape))
        
    return zero_append_list, test_yarray

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
        avg_div = (interval[i - 1] + interval[i]) / 2 
        
        cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2)
        cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2) 
        
        cutted_beat = flatter(record_sg[cut_pre_add:cut_post_add])
        if check_ann in NORAML_ANN:
            record_ann_sym[i] = "N"
            sigN.append(cutted_beat)
            annN.append(record_ann_sym[i])

        elif check_ann in SUPRA_ANN:
            record_ann_sym[i] = "S"
            sigS.append(cutted_beat)
            annS.append(record_ann_sym[i])

        elif check_ann in VENTRI_ANN:
            record_ann_sym[i] = "V"
            sigV.append(cutted_beat)
            annV.append(record_ann_sym[i])

        elif check_ann in FUSION_ANN:
            record_ann_sym[i] = "F"
            sigF.append(cutted_beat)
            annF.append(record_ann_sym[i])

        elif check_ann in UNCLASS_ANN:
            record_ann_sym[i] = "Q"
            sigQ.append(cutted_beat)
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
# Normal beat
R_PATH = DEFAULT_PATH + DB_PATH[0] + "/"

Xtr_N, Ytr_N = makeTrainSet(Xtr_N, Ytr_N, R_PATH)
Xte_N, Yte_N = makeTestSet(Xte_N, Yte_N, R_PATH)

print("- "*15 + "Random Zero-padding applied" + " -" * 15)
print("[SIZE]\t\tXtr_N : {}\t\t\tXte_N : {}\n\t\tYtr_N : {}\t\t\tYte_N : {}".format(Xtr_N.shape, Xte_N.shape, Ytr_N.shape, Yte_N.shape))
print("- "*55)


# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xtr_N))):
#     temp_rpath = R_PATH + str(j)
#     temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

#     windowed_list = Xtr_N[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if np.sum(windowed_list) == 0:
#         continue

#     if len(windowed_list) > 428: 
#         cut_it_off = 0            
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         random_pre_add = np.random.randint(0, 429 - len(windowed_list))
#         random_post_add = 428 - (random_pre_add + len(windowed_list))
    
#         if random_post_add > 428:
#             random_post_add = 427 - random_post_add

#         windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
#     windowed_list = np.array(windowed_list, dtype=np.float64)
#     zero_padded_list.append(windowed_list[:428])
# Xtr_N = np.array(zero_padded_list)

# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xte_N))):
#     # Got R-R Peak by rdann funciton
#     windowed_list = Xte_N[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if len(windowed_list) > 428: 
#         cut_it_off = 0
#         zero_padded_list.append(windowed_list[:428])
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
#             zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
#         else:
#             zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
    
# Xte_N = np.array(zero_padded_list)

# print("- "*15 + "Random Zero-padding applied" + " -" * 15)
# print("[SIZE]\t\tXtr_N : {}\t\t\tXte_N : {}\n\t\tYtr_N : {}\t\t\tYte_N : {}".format(Xtr_N.shape, Xte_N.shape, Ytr_N.shape, Yte_N.shape))
# print("- "*55)

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # Random zero-padding part
# # Supra-Ventricular beat
# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xtr_S))):
#     temp_rpath = R_PATH + str(j)
#     temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

#     windowed_list = Xtr_S[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if np.sum(windowed_list) == 0:
#         continue

#     if len(windowed_list) > 428: 
#         cut_it_off = 0            
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         random_pre_add = np.random.randint(0, 429 - len(windowed_list))
#         random_post_add = 428 - (random_pre_add + len(windowed_list))
    
#         if random_post_add > 428:
#             random_post_add = 427 - random_post_add

#         windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
#     windowed_list = np.array(windowed_list, dtype=np.float64)
#     zero_padded_list.append(windowed_list[:428])
# Xtr_S = np.array(zero_padded_list)

# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xte_S))):
#     # Got R-R Peak by rdann funciton
#     windowed_list = Xte_S[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if len(windowed_list) > 428: 
#         cut_it_off = 0
#         zero_padded_list.append(windowed_list)
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
#             zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
#         else:
#             zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
# Xte_S = np.array(zero_padded_list)
        
# print("- "*15 + "Random Zero-padding applied" + " -" * 15)
# print("[SIZE]\t\tXtr_S : {}\t\t\tXte_S : {}\n\t\tYtr_S : {}\t\t\tYte_S : {}".format(Xtr_S.shape, Xte_S.shape, Ytr_S.shape, Yte_S.shape))
# print("- "*55)

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # Random zero-padding part
# # Ventricular beat
# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xtr_V))):
#     temp_rpath = R_PATH + str(j)
#     temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

#     windowed_list = Xtr_V[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if np.sum(windowed_list) == 0:
#         continue

#     if len(windowed_list) > 428: 
#         cut_it_off = 0            
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         random_pre_add = np.random.randint(0, 429 - len(windowed_list))
#         random_post_add = 428 - (random_pre_add + len(windowed_list))
    
#         if random_post_add > 428:
#             random_post_add = 427 - random_post_add

#         windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
#     windowed_list = np.array(windowed_list, dtype=np.float64)
#     zero_padded_list.append(windowed_list[:428])
# Xtr_V = np.array(zero_padded_list)

# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xte_V))):
#     # Got R-R Peak by rdann funciton
#     windowed_list = Xte_V[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if len(windowed_list) > 428: 
#         cut_it_off = 0
#         zero_padded_list.append(windowed_list)
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
#             zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
#         else:
#             zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
# Xte_V = np.array(zero_padded_list)
        
# print("- "*15 + "Random Zero-padding applied" + " -" * 15)
# print("[SIZE]\t\tXtr_V : {}\t\t\tXte_V : {}\n\t\tYtr_V : {}\t\t\tYte_V : {}".format(Xtr_V.shape, Xte_V.shape, Ytr_V.shape, Yte_V.shape))
# print("- "*55)

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # Random zero-padding part
# # False beat
# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xtr_F))):
#     temp_rpath = R_PATH + str(j)
#     temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

#     windowed_list = Xtr_F[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if np.sum(windowed_list) == 0:
#         continue

#     if len(windowed_list) > 428: 
#         cut_it_off = 0            
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         random_pre_add = np.random.randint(0, 429 - len(windowed_list))
#         random_post_add = 428 - (random_pre_add + len(windowed_list))
    
#         if random_post_add > 428:
#             random_post_add = 427 - random_post_add

#         windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
#     windowed_list = np.array(windowed_list, dtype=np.float64)
#     zero_padded_list.append(windowed_list[:428])
# Xtr_F = np.array(zero_padded_list)

# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xte_F))):
#     # Got R-R Peak by rdann funciton
#     windowed_list = Xte_F[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if len(windowed_list) > 428: 
#         cut_it_off = 0
#         zero_padded_list.append(windowed_list)
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
#             zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
#         else:
#             zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
# Xte_F = np.array(zero_padded_list)
        
# print("- "*15 + "Random Zero-padding applied" + " -" * 15)
# print("[SIZE]\t\tXtr_F : {}\t\t\tXte_F : {}\n\t\tYtr_F : {}\t\t\tYte_F : {}".format(Xtr_F.shape, Xte_F.shape, Ytr_F.shape, Yte_F.shape))
# print("- "*55)

# # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# # Random zero-padding part
# # Unclassed beat
# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xtr_Q))):
#     temp_rpath = R_PATH + str(j)
#     temp_pickle = PICKLE_PATH + DB_PATH[0] + str(j) + ".pkl"

#     windowed_list = Xtr_Q[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if np.sum(windowed_list) == 0:
#         continue

#     if len(windowed_list) > 428: 
#         cut_it_off = 0            
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         random_pre_add = np.random.randint(0, 429 - len(windowed_list))
#         random_post_add = 428 - (random_pre_add + len(windowed_list))
    
#         if random_post_add > 428:
#             random_post_add = 427 - random_post_add

#         windowed_list = np.append([0.0], np.pad(windowed_list, (random_pre_add, random_post_add) , 'constant', constant_values=0))
    
#     windowed_list = np.array(windowed_list, dtype=np.float64)
#     zero_padded_list.append(windowed_list[:428])
# Xtr_Q = np.array(zero_padded_list)

# dict_ann = []
# windowed_list = []
# zero_padded_list = []
# for j in tqdm(range(len(Xte_Q))):
#     # Got R-R Peak by rdann funciton
#     windowed_list = Xte_Q[j]
#     cut_it_off = int((428 - len(windowed_list)) / 2)

#     if len(windowed_list) > 428: 
#         cut_it_off = 0
#         zero_padded_list.append(windowed_list)
        
#     else:
#         cut_it_off = int((428 - len(windowed_list)) / 2)

#         if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
#             zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
#         else:
#             zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
# Xte_Q = np.array(zero_padded_list)
        
# print("- "*15 + "Random Zero-padding applied" + " -" * 15)
# print("[SIZE]\t\tXtr_Q : {}\t\t\tXte_Q : {}\n\t\tYtr_Q : {}\t\t\tYte_Q : {}".format(Xtr_Q.shape, Xte_Q.shape, Ytr_Q.shape, Yte_Q.shape))
# print("- "*55)

# print(Ytr_F[0])

# Xtr_S = np.array(Xtr_S)
# Xtr_S = Xtr_S[0]

# Ytr_S = np.array(Ytr_S)
# Ytr_S = Ytr_S[0]

# Xtr_F = np.array(Xtr_F)
# Xtr_F = Xtr_F[0]

# Ytr_F = np.array(Ytr_F)
# Ytr_F = Ytr_F[0]

# dfList = ['S','V','F','Q']

# Xtr = pd.DataFrame(data=Xtr_N)

# for dfName in dfList:
#     df = pd.DataFrame(data=globals()["Xtr_{}".format(dfName)])
#     Xtr= pd.concat([Xtr,df], axis=0)
    
# Xte = pd.DataFrame(data=Xte_N)

# for dfName in dfList:
#     df = pd.DataFrame(data=globals()["Xte_{}".format(dfName)])
#     Xte = pd.concat([Xte,df], axis=0)
    
# Ytr, Yte = [],[]

# for i in range(len(Ytr_N)):
#     Ytr.append(0)

# for i in range(len(Ytr_S)):
#     Ytr.append(1) 

# for i in range(len(Ytr_V)):
#     Ytr.append(2)

# for i in range(len(Ytr_F.tolist())):
#     Ytr.append(3)

# for i in range(len(Ytr_Q)):
#     Ytr.append(4)

# Ytr = np.array(Ytr)
# Ytr = pd.DataFrame(data=Ytr)

# for i in range(len(Yte_N)):
#     Yte.append(0)

# for i in range(len(Yte_S)):
#     Yte.append(1) 

# for i in range(len(Yte_V)):
#     Yte.append(2)

# for i in range(len(Yte_F)):
#     Yte.append(3)

# for i in range(len(Yte_Q)):
#     Yte.append(4)

# Yte = np.array(Yte)
# Yte = pd.DataFrame(data=Yte)

# X_train = np.array(Xtr[list(range(428))].values)[..., np.newaxis]
# Y_train = np.array(Ytr[0].values).astype(np.int8)
# X_val = np.array(Xte[list(range(428))].values)[..., np.newaxis]
# Y_val = np.array(Yte[0].values).astype(np.int8)

# oneHot = LabelEncoder()
# oneHot.fit(Y_train)
# oneHot.fit(Y_val)
# Y_train = oneHot.transform(Y_train)
# Y_val = oneHot.transform(Y_val)

# X_train = X_train.reshape(-1, 428, 1)
# X_val = X_val.reshape(-1, 428, 1)
# Y_train = to_categorical(Y_train, 5)
# Y_val = to_categorical(Y_val, 5)
                       
# Ytr = pd.DataFrame(data=Y_train)
# Yte = pd.DataFrame(data=Y_val)
# dfY = pd.concat([Ytr,Yte], axis=0)

# print("X_train shape: ", X_train.shape)
# print("Y_trainshape: ", Y_train.shape)
# print("──────────────────────────")
# print("X_test shape: ", X_val.shape)
# print("Y_test shape: ", Y_val.shape)
# print("──────────────────────────")
# print(dfY.value_counts())