from tabnanny import check
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle

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