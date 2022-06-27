from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import random
import pickle
import os

DATA_PATH = "./pickle_mat/"
TEST_SIZE = 0.3
RANDOM_STATE = 42


record_list = []
pickle_input = dict()
X, y = [], []

print("[INFO] Read records file from ", DATA_PATH)
with open(DATA_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

normal_sig_list = list()
ven_sig_list = list()
supra_sig_list = list()
false_sig_list = list()
unclass_sig_list = list()

normal_ann_list = list()
ven_ann_list = list()
supra_ann_list = list()
false_ann_list = list()
unclass_ann_list = list()

for i in tqdm(range(len(record_list))):
    temp_path = DATA_PATH + "mitPssigP_" + record_list[i] + ".pkl"
    with open(temp_path, 'rb') as f:
        pickle_input = pickle.load(f)

        for i in range(len(pickle_input[0])):
            check_ann = pickle_input[1][i]
    
            if check_ann == "N":            # Normal
                normal_sig_list.append(pickle_input[0][i])
                normal_ann_list.append(0)

            elif check_ann == "S":          # Supra-ventricular
                supra_sig_list.append(pickle_input[0][i])
                supra_ann_list.append(1)

            elif check_ann == "V":
                ven_sig_list.append(pickle_input[0][i])
                ven_ann_list.append(2)

            elif check_ann == "F":          # False alarm
                false_sig_list.append(pickle_input[0][i])
                false_ann_list.append(3)

            elif check_ann == "Q":          # Unclassed 
                unclass_sig_list.append(pickle_input[0][i])
                unclass_ann_list.append(4)
        
            else:
                pass

# Size check
np_normal_sig = np.array(normal_sig_list)
np_normal_ann = np.array(normal_ann_list)

np_supra_sig = np.array(supra_sig_list)
np_supra_ann = np.array(supra_ann_list)

np_ven_sig = np.array(ven_sig_list)
np_ven_ann = np.array(ven_ann_list)

np_false_sig = np.array(false_sig_list)
np_false_ann = np.array(false_ann_list)

np_unclass_sig = np.array(unclass_sig_list)
np_unclass_ann = np.array(unclass_ann_list)

print("[SIZE]\t\tNormal Beat Signal : {}\n\t\tNormal Annotation : {}".format(np_normal_sig.shape, np_normal_ann.shape))
print("[SIZE]\t\tSupra-ventricular Beat Signal : {}\n\t\tSupra-ventricular Annotation : {}".format(np_supra_sig.shape, np_supra_ann.shape))
print("[SIZE]\t\tVentricular Beat Signal : {}\n\t\tVentricular Annotation : {}".format(np_ven_sig.shape, np_ven_ann.shape))
print("[SIZE]\t\tFalse Beat Signal : {}\n\t\tFalse Annotation : {}".format(np_false_sig.shape, np_false_ann.shape))
print("[SIZE]\t\tUnclassed Beat Signal : {}\n\t\tUnclassed Annotation : {}".format(np_unclass_sig.shape, np_unclass_ann.shape))

print("\n[INFO] Normal Beat * Down-sizing *")

# Random choice normal beats
normal_random_sig = random.choices(normal_sig_list, k=10000)
normal_random_ann = random.choices(normal_ann_list, k=10000)
np_normal_sig = np.array(normal_random_sig)
np_normal_ann = np.array(normal_random_ann)
print("[SIZE]\t\tNormal Beat Signal : {}\n\t\tNormal Annotation : {}\n".format(np_normal_sig.shape, np_normal_ann.shape))

# Train test split part
X_train_normal, X_test_normal, y_train_normal, y_test_normal = train_test_split(np_normal_sig, np_normal_ann,       test_size=0.3, random_state=42, shuffle=True)
X_train_supra, X_test_supra, y_train_supra, y_test_supra = train_test_split(np_supra_sig, np_supra_ann,             test_size=0.3, random_state=42, shuffle=True)
X_train_ven, X_test_ven, y_train_ven, y_test_ven = train_test_split(np_ven_sig, np_ven_ann,                         test_size=0.3, random_state=42, shuffle=True)
X_train_false, X_test_false, y_train_false, y_test_false = train_test_split(np_false_sig, np_false_ann,             test_size=0.3, random_state=42, shuffle=True)
X_train_unclass, X_test_unclass, y_train_unclass, y_test_unclass = train_test_split(np_unclass_sig, np_unclass_ann, test_size=0.3, random_state=42, shuffle=True)

print("$ " * 13 + "TRAIN TEST SPLIT " + "$ " * 13)
print("[SIZE]\t\tTrain Normal Beat Signal : {}\n\t\tTrain Normal Annotation : {}".format(X_train_normal.shape, y_train_normal.shape))
print("[SIZE]\t\tTest Normal Beat Signal : {}\n\t\tTest Normal Annotation : {}".format(X_test_normal.shape, y_test_normal.shape))

print("= " * 35)
print("[SIZE]\t\tTrain Supra-ventricular Beat Signal : {}\n\t\tTrain Supra-ventricular Annotation : {}".format(X_train_supra.shape, y_train_supra.shape))
print("[SIZE]\t\tTest Supra-ventricular Beat Signal : {}\n\t\tTest Supra-ventricular Annotation : {}".format(X_test_supra.shape, y_test_supra.shape))

print("= " * 35)
print("[SIZE]\t\tTrain Ventricular Beat Signal : {}\n\t\tTrain Ventricular Annotation : {}".format(X_train_ven.shape, y_train_ven.shape))
print("[SIZE]\t\tTest Ventricular Beat Signal : {}\n\t\tTest Ventricular Annotation : {}".format(X_test_ven.shape, y_test_ven.shape))

print("= " * 35)
print("[SIZE]\t\tTrain False Beat Signal : {}\n\t\tTrain False Annotation : {}".format(X_train_false.shape, y_train_false.shape))
print("[SIZE]\t\tTest False Beat Signal : {}\n\t\tTest False Annotation : {}".format(X_test_false.shape, y_test_false.shape))

print("= " * 35)
print("[SIZE]\t\tTrain Unclassed Beat Signal : {}\n\t\tTrain Unclassed Annotation : {}".format(X_train_unclass.shape, y_train_unclass.shape))
print("[SIZE]\t\tTest Unclassed Beat Signal : {}\n\t\tTest Unclassed Annotation : {}".format(X_test_unclass.shape, y_test_unclass.shape))

# Up-sizing S class
print("= " * 35)
print("\n"*3)
print("[INFO] Supra-Venticular classes starting \"Rebalancing\"...")
sizing_s_class_sig = list()
sizing_s_class_ann = list()

for idx in tqdm(range(5000)):
    sizing_s_class_sig.append(random.choice(X_train_supra))
    sizing_s_class_ann.append(random.choice(y_train_supra))

sizing_s_class_sig = np.array(sizing_s_class_sig)
sizing_s_class_ann = np.array(sizing_s_class_ann)
print("[SIZE]\t\tRe-balanced Supra-ventricular Beat Signal : {}\n\t\tRe-balanced Supra-ventricular Annotation : {}".format(sizing_s_class_sig.shape, sizing_s_class_ann.shape))

# Up-sizing F class
print("\n"*3)
print("[INFO] False alarm classes starting \"Rebalancing\"...")
sizing_f_class_sig = list()
sizing_f_class_ann = list()

for idx in tqdm(range(8000)):
    sizing_f_class_sig.append(random.choice(X_train_false))
    sizing_f_class_ann.append(random.choice(y_train_false))

sizing_f_class_sig = np.array(sizing_f_class_sig)
sizing_f_class_ann = np.array(sizing_f_class_ann)
print("[SIZE]\t\tRe-balanced False Beat Signal : {}\n\t\tRe-balanced False Annotation : {}".format(sizing_f_class_sig.shape, sizing_f_class_ann.shape))