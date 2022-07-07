
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import itertools
import pickle
import random

EPOCH = 200
KERNEL_SIZE = 3
POOLING_SIZE = 2
BATCH_SIZE = 128

DATA_PATH = "./pickle_rand/"

def list_to_list(input_list):
    input_list_to_list = list(itertools.chain(*input_list))
    return input_list_to_list

# Dataload part
le = preprocessing.LabelEncoder()

record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

for i in tqdm(range(len(record_list))):
    temp_path = DATA_PATH + "mit" + record_list[i] + ".pkl"
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

            y.append(temp_ann_list)

# 데이터 갯수 파악
# N, S, V, F, Q
uni, cnt = np.unique(np.array(y), return_counts=True)
print(uni, cnt)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42, shuffle=True )

npx = np.array(X_train, dtype=object)
npy = np.array(y_train)
npx_vali = np.array(X_val, dtype=object)
npy_vali = np.array(y_val)
npx_test = np.array(X_test, dtype=object)
npy_test = np.array(y_test)

print("[SIZE]\t\tNpX lenght : {}\n\t\tNpY length : {}".format(npx.shape, npy.shape))
print("[SIZE]\t\tX_validation length : {}\n\t\ty_validation length : {}".format(npx_vali.shape, npy_vali.shape))
print("[SIZE]\t\tX_test length : {}\n\t\ty_test length : {}".format(npx_test.shape, npy_test.shape))

# 랜덤으로 뽑아서 뿌려보기
random.seed(64)

plt.figure(figsize=(30, 12))
plt.suptitle("ECG Signal random padding", fontsize=18)
n = 0

for i in random.sample(range(6480), 60):
    ax = plt.subplot(8, 8, n+1)
    plt.plot(npx[i])
    ax.set_title(str(npy[i]))
    n+=1

plt.show()