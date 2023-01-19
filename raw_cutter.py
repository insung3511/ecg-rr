# Cutting as a beat that just raw signal.
# That's it.

from wfdb import processing
from wfdb import rdsamp
from wfdb import rdann

from scipy import signal

import matplotlib.pyplot as plt
import wfdb.processing as wp
import numpy as np
import pickle
import wfdb
import os

PICKLE_PATH = "./pickle/"
DEFAULT_PATH = "./data/"
DB_PATH = ['mit']
EXTRA_NP = np.array(0)

NORAML_ANN = ['N', 'L', 'R', 'e', 'j']
SUPRA_ANN = ['A', 'a', 'J', 'S']
VENTRI_ANN = ['V', 'E']
FUSION_ANN = ['F']
UNCLASS_ANN = ['/', 'f', 'Q']

WINDOW_SIZE = 265

class Filters():
    def lowpass_filter(x, low, order=6, fs=360):
        nyq = 0.5 * fs
        normal_cutoff = low / nyq
        b, a = signal.butter(order, normal_cutoff, btype='low')
        filtered_x = signal.lfilter(b, a, x)
        return filtered_x

    def highpass_filter(x, high, order=6, fs=360):
        nyq = 0.5 * fs
        normal_cutoff = high / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high')
        filtered_x = signal.lfilter(b, a, x)
        return filtered_x

    def bandpass_filter(x, low, high, order=6, fs=360):
        nyq = 0.5 * fs
        low, high = low / nyq, high / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        filtered_x = signal.lfilter(b, a, x)
        return filtered_x


def flatter(list_of_list):
    flatList = [ item for elem in list_of_list for item in elem]
    return flatList

for k in range(len(DB_PATH)):
    R_PATH = DEFAULT_PATH + DB_PATH[k] + "/"

    exclude_record = ["bw", "em", "ma"]

    dict_ann = []
    windowed_list = []
    record_list = []
    record_ann = []
    longest = 0.

    # Read RECORDS txt file
    print("[INFO] Read records file from ", R_PATH)
    with open(R_PATH + 'RECORDS') as f:
        record_lines = f.readlines()

    # Read Records
    for i in range(len(record_lines)):
        if record_lines[i].strip() in exclude_record:
            continue
        record_list.append(str(record_lines[i].strip()))


    for j in range(len(record_list)):
        zero_padded_list = []
        dict_ann = []
        temp_rpath = R_PATH + record_list[j]
        temp_pickle = PICKLE_PATH + DB_PATH[k] + record_list[j] + ".pkl"

        # Read original sample by rdsamp function
        record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[0], sampfrom=0)
        record_sg = wp.normalize_bound(record_sg)

        record_sg = wp.normalize_bound(record_sg)
        record_sg = Filters.lowpass_filter(record_sg, 100)
        record_sg = Filters.highpass_filter(record_sg, 0.1)
        record_sg = Filters.bandpass_filter(record_sg, 59.5, 60.5)

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
            elif check_ann in SUPRA_ANN:
                record_ann_sym[i] = "S"
            elif check_ann in VENTRI_ANN:
                record_ann_sym[i] = "V"
            elif check_ann in FUSION_ANN:
                record_ann_sym[i] = "F"
            elif check_ann in UNCLASS_ANN:
                record_ann_sym[i] = "Q"
            else:
                record_ann_sym[i] = " "
            
            avg_div = (interval[i - 1] + interval[i]) / 2 
            cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2)
            cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2) 

            windowed_list = flatter(record_sg[cut_pre_add:cut_post_add])

            cut_it_off = int((WINDOW_SIZE - len(windowed_list)) / 2)

            if len(windowed_list) > WINDOW_SIZE: 
                cut_it_off = 0
                
                cut_pre_add = record_ann[i] - int(WINDOW_SIZE / 2)
                cut_post_add = record_ann[i] + int(WINDOW_SIZE / 2) 
                windowed_list = flatter(record_sg[cut_pre_add:cut_post_add])
                zero_padded_list.append(windowed_list)
                
            else:
                cut_it_off = int((WINDOW_SIZE - len(windowed_list)) / 2)

                if len(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0)) == 427:
                    zero_padded_list.append(np.append([0.0], np.pad(windowed_list, cut_it_off , 'constant', constant_values=0)))
                else:
                    zero_padded_list.append(np.pad(windowed_list, cut_it_off, 'constant', constant_values=0))
              
            dict_ann.append(record_ann_sym[i])
            
            if record_ann_sym[i] != 'N':
                plt.plot(zero_padded_list[-1])
                plt.title(record_ann_sym[i])
                plt.show()

        ann_dict = {
            0 : zero_padded_list,
            1 : dict_ann
        }

        with open(temp_pickle, "wb") as f:
            pickle.dump(ann_dict, f)
        print(temp_pickle, " SAVED!")