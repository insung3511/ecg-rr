from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy import io
import pandas as pd
import numpy as np
import random
import pickle
import os

path = './pickle_mat/'
rList = []

# Read RECORDS txt file
print("[INFO] Read records file from ", path)
with open(path + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    rList.append(str(record_lines[i].strip()))

pickle_input = dict()
X, y = [], []

sigN, sigV, sigS, sigF, sigQ = [],[],[],[],[]
annN, annV, annS, annF, annQ = [],[],[],[],[]

for i in tqdm(range(len(rList))):
    temp_path = path + "mitPssigP_" + rList[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)

        for i in range(len(pickle_input[0])):
            check_ann = pickle_input[1][i]
    
            if check_ann == "N":
                sigN.append(pickle_input[0][i])
                annN.append(0)

            elif check_ann == "S":
                sigS.append(pickle_input[0][i])
                annS.append(1)

            elif check_ann == "V":
                sigV.append(pickle_input[0][i])
                annV.append(2)

            elif check_ann == "F":
                sigF.append(pickle_input[0][i])
                annF.append(3)

            elif check_ann == "Q":
                sigQ.append(pickle_input[0][i])
                annQ.append(4)
        
            else:
                pass

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

print("-"*35)

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

