import wfdb.processing as wp
import numpy as np
import pickle
import wfdb

R_PATH = "./data/mit/"
DB_NAME = "mit"
exclude_record = ["bw", "em", "ma"]

zero_padded_list = []
dict_ann = []
windowed_list = []
record_list = []
record_ann = []
longest = 0.

def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

# Read RECORDS txt file
print("[INFO] Read records file from ", R_PATH)
with open(R_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

# Read Records
for i in range(len(record_lines)):
    if record_lines[i].strip() in exclude_record:
        continue
    record_list.append(str(record_lines[i].strip()))

# Find Longest R-R Interval for window
for i in range(len(record_list)):
    temp_rpath = R_PATH + record_list[i]
    interval = wp.ann2rr(temp_rpath, 'atr', as_array=True)
    longest = interval.max()
print("LONGEST: ", longest)

for i in range(len(record_list)):
    temp_rpath = R_PATH + record_list[i]
    temp_pickle = "./pickle/" + DB_NAME + record_list[i] + ".pkl"

    # Read original sample by rdsamp function
    record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[0], sampfrom=0)

    # Got R-R Peak by rdann funciton
    record_ann = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).sample)[1:]
    record_ann_sym = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).symbol)[1:]

    for i in range(len(record_ann)):
        try:
            pre_add = record_ann[i - 1]
            post_add = record_ann[i + 1]
        except IndexError:
            break

        cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2) 
        cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2)

        # print(pre_add, post_add, "\t\t\t", cut_pre_add, cut_post_add)

        if i < 1:
            continue
        
        windowed_list = flatter(record_sg[cut_pre_add:cut_post_add])
        # print(windowed_list)

        zero_padded_list.append(np.pad(windowed_list, int((longest - len(windowed_list)) / 2), 'constant', constant_values=0))
        dict_ann.append(record_ann_sym[i])

    ann_dict = {
        0 : zero_padded_list,
        1 : dict_ann
    }

    with open(temp_pickle, "wb") as f:
        pickle.dump(ann_dict, f)
    print(temp_pickle, " SAVED!")