from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import os

rList = ["100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
         "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
         "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
         "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
         "222", "223", "228", "230", "231", "232", "233", "234"]
path = './pickle_mat/'
pickle_input = dict()
X, y = [], []

sigN, sigV, sigS, sigF, sigQ = [], [], [], [], []
annN, annV, annS, annF, annQ = [], [], [], [], []

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

sigN_ran = random.sample(sigN, 10000)
annN_ran = random.sample(annN, 10000)
sigN_np = np.array(sigN_ran)
annN_np = np.array(annN_ran)

Xtr_N, Xte_N, Ytr_N, Yte_N = train_test_split(sigN_np, annN_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_S, Xte_S, Ytr_S, Yte_S = train_test_split(sigS_np, annS_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_V, Xte_V, Ytr_V, Yte_V = train_test_split(sigV_np, annV_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_F, Xte_F, Ytr_F, Yte_F = train_test_split(sigF_np, annF_np, test_size=0.3, random_state=42, shuffle=True)
Xtr_Q, Xte_Q, Ytr_Q, Yte_Q = train_test_split(sigQ_np, annQ_np, test_size=0.3, random_state=42, shuffle=True)

sigS_ran, annS_ran, sigF_ran, annF_ran =[],[],[],[]

sigS_ran.append(random.choices(Xtr_S, k=5000))
annS_ran.append(random.choices(Ytr_S, k=5000))
sigF_ran.append(random.choices(Xtr_F, k=8000))
annF_ran.append(random.choices(Ytr_F, k=8000))

Xtr_S = np.array(sigS_ran)
Xtr_S = Xtr_S[0][:][:]
Ytr_S = np.array(annS_ran)
Ytr_S = Ytr_S[0][:][:]
Xtr_F = np.array(sigF_ran)
Xtr_F = Xtr_F[0][:][:]
Ytr_F = np.array(annF_ran)
Ytr_F = Xtr_F[0][:][:]

dfList = ['S','V','F','Q']

Xtr = pd.DataFrame(data=Xtr_N)

for dfName in dfList:
    df = pd.DataFrame(data=globals()["Xtr_{}".format(dfName)])
    Xtr= pd.concat([Xtr,df], axis=0)
    
Xte = pd.DataFrame(data=Xte_N)

for dfName in dfList:
    df = pd.DataFrame(data=globals()["Xte_{}".format(dfName)])
    Xte = pd.concat([Xte,df], axis=0)
    
Ytr, Yte = [],[]

for i in range(0,7000):
    Ytr.append(0)

for i in range(0,5000):
    Ytr.append(1) 

for i in range(0,5063):
    Ytr.append(2)

for i in range(0,8000):
    Ytr.append(3)

for i in range(0,5627):
    Ytr.append(4)

Ytr = np.array(Ytr)
Ytr = pd.DataFrame(data=Ytr)

for i in range(0,3000):
    Yte.append(0)
for i in range(0,835):
    Yte.append(1) 
for i in range(0,2171):
    Yte.append(2)
for i in range(0,241):
    Yte.append(3)
for i in range(0,2412):
    Yte.append(4)

Yte = np.array(Yte)
Yte = pd.DataFrame(data=Yte)

X_train = np.array(Xtr[list(range(428))].values)[..., np.newaxis]
Y_train = np.array(Ytr[0].values).astype(np.int8)
X_val = np.array(Xte[list(range(428))].values)[..., np.newaxis]
Y_val = np.array(Yte[0].values).astype(np.int8)

oneHot = LabelEncoder()
oneHot.fit(Y_train)
oneHot.fit(Y_val)
Y_train = oneHot.transform(Y_train)
Y_val = oneHot.transform(Y_val)

X_train = X_train.reshape(-1, 428, 1)
X_val = X_val.reshape(-1, 428, 1)
Y_train = to_categorical(Y_train, 5)
Y_val = to_categorical(Y_val, 5)
                       
Ytr = pd.DataFrame(data=Y_train)
Yte = pd.DataFrame(data=Y_val)
dfY = pd.concat([Ytr,Yte], axis=0)

print("X_train shape: ", X_train.shape)
print("Y_trainshape: ", Y_train.shape)
print("──────────────────────────")
print("X_test shape: ", X_val.shape)
print("Y_test shape: ", Y_val.shape)
print("──────────────────────────")
print(dfY.value_counts())