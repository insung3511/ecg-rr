# Cutting as a beat that filtered by Ubin.
from turtle import shape
import matplotlib.pyplot as plt
import scipy.io as io
import numpy as np
import itertools
import random

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

def padding_remover(array):
    removed_array = []
    array = list(array)

    for beat_array in array:
        removed_beat_array = []
        for beat_element in beat_array:
            if beat_element == 0 or beat_element == 1 or beat_element == 2 or beat_element == 3 or beat_element == 4:
                continue

            else:
                removed_beat_array.append(beat_element)

        removed_array.append(removed_beat_array)
    return removed_array

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

def concater(normal, supra, ventri, fusion, q):
    return list(itertools.chain(normal, supra, ventri, fusion, q))

def shape_check(title, x, y):
    plt.figure(figsize=(30, 12))
    plt.suptitle(title, fontsize=18)
    n = 0
    
    for i in random.sample(range(len(x)), 25):
        ax = plt.subplot(5, 5, n+1)
        plt.plot(x[i])
        ax.set_title(str(y[i]))

        n+=1
    plt.show()
    plt.clf()

print("[INFO] Training set start to padding. **RANDOM PADDING**")
path = './data/fix_error/Rs2Tr.mat'
X, y = [], []

x = io.loadmat(path)
x = x['data']

sigN = x[:7000]
sigS = x[7000:8947]
sigV = x[8947:14011]
sigF = x[14011:14582]
sigQ = x[14582:20209]

annN, annV, annS, annF, annQ = [], [], [], [], []

sigN = padding_remover(sigN)
sigN_np = np.array(sigN)
annN_np = np.array(annN)

sigS = padding_remover(sigS)
sigS_np = np.array(sigS)
annS_np = np.array(annS)

sigV = padding_remover(sigV)
sigV_np = np.array(sigV)
annV_np = np.array(annV)

sigF = padding_remover(sigF)
sigF_np = np.array(sigF)
annF_np = np.array(annF)

sigQ = padding_remover(sigQ)
sigQ_np = np.array(sigQ)
annQ_np = np.array(sigQ)

sigS_ran, annS_ran, sigF_ran, annF_ran =[],[],[],[]

# - - - - - - - - - - - - - - - - - - - - -
# Under-sampling
# - - - - - - - - - - - - - - - - - - - - -
sigN_ran = []
annN_ran = []
for sig in range(7000):
    sigN_ran.append(random.choice(sigN_np))
    annN_ran.append(0)                      # Normal
Xtr_N = np.array(sigN_ran)
Ytr_N = np.array(annN_ran)

# - - - - - - - - - - - - - - - - - - - - -
# Over-sampling
# - - - - - - - - - - - - - - - - - - - - -
sigV_ran = []
annV_ran = []
for sig in range(7000):
    sigV_ran.append(random.choice(sigV_np))
    annV_ran.append(1)                      # Ventricular
Xtr_V = np.array(sigV_ran)
Ytr_V = np.array(annV_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigS_ran = []
annS_ran = []
for sig in range(7000):
    sigS_ran.append(random.choice(sigS_np))
    annS_ran.append(2)                      # Supra-Ventricular
Xtr_S = np.array(sigS_ran)
Ytr_S = np.array(annS_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigF_ran = []
annF_ran = []
for sig in range(7000):
    sigF_ran.append(random.choice(sigF_np))
    annF_ran.append(3)                      # Fusion
Xtr_F = np.array(sigF_ran)
Ytr_F = np.array(annF_ran)

# - - - - - - - - - - - - - - - - - - - - -
sigQ_ran = []
annQ_ran = []
for sig in range(7000):
    sigQ_ran.append(random.choice(sigQ_np))
    annQ_ran.append(4)                      # Unkownu
Xtr_Q = np.array(sigQ_ran)
Ytr_Q = np.array(annQ_ran)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("- "*35)
Xtr_N, Ytr_N = TrainSetPadding(Xtr_N, Ytr_N)
print("[SIZE]\t\tXtr_N : {}\t\tYte_N : {}".format(Xtr_N.shape, Ytr_N.shape))
print("- "*35)

Xtr_S, Ytr_S = TrainSetPadding(Xtr_S, Ytr_S)
print("[SIZE]\t\tXtr_S : {}\t\tYte_S : {}".format(Xtr_S.shape, Ytr_S.shape))
print("- "*35)

Xtr_V, Ytr_V = TrainSetPadding(Xtr_V, Ytr_V)
print("[SIZE]\t\tXtr_V : {}\t\tYte_V : {}".format(Xtr_V.shape, Ytr_V.shape))
print("- "*35)

Xtr_F, Ytr_F = TrainSetPadding(Xtr_F, Ytr_F)
print("[SIZE]\t\tXtr_F : {}\t\tYte_F : {}".format(Xtr_F.shape, Ytr_F.shape))
print("- "*35)

Xtr_Q, Ytr_Q = TrainSetPadding(Xtr_Q, Ytr_Q)
print("[SIZE]\t\tXtr_Q : {}\t\tYte_Q : {}".format(Xtr_Q.shape, Ytr_Q.shape))
print("- "*35 + "\n")

print("- "*10 + "Final Train concat split result " + "- "*10)
X_train = np.array(concater(Xtr_N, Xtr_S, Xtr_V, Xtr_F, Xtr_Q))
y_train = np.array(concater(Ytr_N, Ytr_S, Ytr_V, Ytr_F, Ytr_Q))
print("[SIZE]\t\tX_train : {}\t\ty_train : {}".format(X_train.shape, y_train.shape))
print("- "*35)

shape_check("Train shape check", X_train, y_train)

#
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
#

print("\n\n\a[INFO] Validation set start to padding. **Standrad PADDING**")
path = './data/fix_error/Rs2Tr.mat'
X, y = [], []

x = io.loadmat(path)
x = x['data']

sigN = x[:3000]
sigS = x[3000:3834]
sigV = x[3834:6004]
sigF = x[6004:6245]
sigQ = x[6245:8657]

annN, annV, annS, annF, annQ = [], [], [], [], []

sigN = padding_remover(sigN)
sigN_np = np.array(sigN)
annN_np = np.array(annN)

sigS = padding_remover(sigS)
sigS_np = np.array(sigS)
annS_np = np.array(annS)

sigV = padding_remover(sigV)
sigV_np = np.array(sigV)
annV_np = np.array(annV)

sigF = padding_remover(sigF)
sigF_np = np.array(sigF)
annF_np = np.array(annF)

sigQ = padding_remover(sigQ)
sigQ_np = np.array(sigQ)
annQ_np = np.array(sigQ)

for sig in range(len(sigN)):
    annN.append(0)              # Normal
Yte_N = np.array(annN_ran)

for sig in range(len(sigS)):
    annS.append(1)              # Supra-Ventricular
Yte_S = np.array(annS_ran)

for sig in range(len(sigV)):
    annV.append(2)              # Ventricular
Yte_V = np.array(annV_ran)

for sig in range(len(sigF)):
    annF.append(3)              # Fusion
Yte_F = np.array(annF_ran)

for sig in range(len(sigQ)):
    annQ.append(4)              # Unclassed
Yte_Q = np.array(annQ_ran)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
print("- "*35)
Xte_N, Yte_N = TestSetPadding(sigN_np, Yte_N)
print("[SIZE]\t\tXte_N : {}\t\tYte_N : {}".format(Xte_N.shape, Yte_N.shape))
print("- "*35)

Xte_S, Yte_S = TestSetPadding(sigS_np, Yte_S)
print("[SIZE]\t\tXte_S : {}\t\tYte_S : {}".format(Xte_S.shape, Yte_S.shape))
print("- "*35)

Xte_V, Yte_V = TestSetPadding(sigV_np, Yte_V)
print("[SIZE]\t\tXte_V : {}\t\tYte_V : {}".format(Xte_V.shape, Yte_V.shape))
print("- "*35)

Xte_F, Yte_F = TestSetPadding(sigF_np, Yte_F)
print("[SIZE]\t\tXte_F : {}\t\tYte_F : {}".format(Xte_F.shape, Yte_F.shape))
print("- "*35)

Xte_Q, Yte_Q = TestSetPadding(sigQ_np, Yte_Q)
print("[SIZE]\t\tXte_Q : {}\t\tYte_Q : {}".format(Xte_Q.shape, Yte_Q.shape))
print("- "*35 + "\n")

print("- "*10 + "Final Test concat split result " + "- "*10)
X_test = np.array(concater(Xte_N, Xte_S, Xte_V, Xte_F, Xte_Q))
y_test = np.array(concater(Yte_N, Yte_S, Yte_V, Yte_F, Yte_Q))
print("[SIZE]\t\tX_test : {}\t\ty_test : {}".format(X_test.shape, y_test.shape))
print("- "*35)

shape_check("Test shape check", X_test, y_test)