from os import pread
import matplotlib.pyplot as plt
from sympy import Q
import wfdb.processing as wp
import wfdb

R_PATH = "./data/mit/"
exclude_record = ['bw', 'em', 'ma']

zero_padded_list = []
record_list = []
record_avge = []
longest = 0.0

print("[INFO] Read records file from ", R_PATH)
with open(R_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

# Read Records
for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))
print(record_list)

temp_sasg = []
cnt_ecg = 0

# R-R Interval
for i in range(len(record_list)):
    temp_avge_cnt = 0
    target_list = []
    left_size, right_size = 0, 0

    if record_list[i] in exclude_record:
        continue

    temp_path = R_PATH + record_list[i]
    temp_rr = (wp.ann2rr(temp_path, 'atr', as_array=True, start_time=0.0))
    print(temp_rr.tolist())
    temp_sg, temp_fs = wfdb.rdsamp(temp_path, channels=[0, 1], sampfrom=0)

    for id, i in enumerate(range(len(temp_rr))):  # Might be window size
        try:
            left_size += temp_rr[id]
            right_size += temp_rr[id + 1]
        except IndexError:
            pass

        diff_len = right_size - left_size
        print(left_size, right_size)

        plt.figure(figsize=(15, 4))
        plt.plot(temp_sg)
        plt.plot(temp_sg[left_size+left_size:right_size+right_size])
        plt.show()
        
        if id == temp_rr[i]:
            print("SAME!")
