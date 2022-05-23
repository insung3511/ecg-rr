from os import pread
import matplotlib.pyplot as plt
from sympy import Q
import wfdb.processing as wp
import wfdb

R_PATH = "./data/nstdb/"

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

    temp_path = R_PATH + record_list[i]
    temp_rr = (wp.ann2rr(temp_path, 'atr', as_array=True, start_time=0.0))

    temp_sg, temp_fs = wfdb.rdsamp(temp_path, sampfrom=0)
    xqrs = wp.XQRS(sig=temp_sg[:, 0], fs=temp_fs['fs'])
    xqrs.detect()
    
    temp_cnt = -len(temp_rr)

    for id, i in enumerate(range(len(temp_rr))):  # Might be window size
        avge_result = (temp_rr[temp_cnt + 1] + temp_rr[(temp_cnt + 2)]) / 2   
        record_avge.append(avge_result)                                     # Save it...
        temp_cnt += 1                       
        
        if id == temp_rr[i]:
            print("SAME!")

        if id < 2:
            continue
             
        print(cnt_ecg, cnt_ecg + cnt_ecg)
        cnt_ecg += int(xqrs.rr_init)

        temp_sasg = (temp_sg[cnt_ecg : cnt_ecg * 2].tolist())
        if id == 1:
            temp_sasg = (temp_sg[0 : 
                                 cnt_ecg].tolist())

        plt.plot(temp_sasg)
        plt.show()
    # plt.show()
        # print(temp_sasg.tolist(), "\n\n\n")

# print(temp_rr.tolist())

# print("Longest strength : {},\tLongest Average : {}".format(longest, longest / 2))

# for i in range(len(record_list)):
#     temp_path = R_PATH + record_list[i]

#     temp_list, _ = list(wfdb.rdsamp(temp_path, sampfrom=0))
#     temp_rr = (wp.ann2rr(temp_path, 'atr', as_array=True, start_time=0.0)).tolist()
    
#     for i in range(len(temp_list)):
#         # zero_padded_list = [
#         #     temp_list[temp_rr[i] - avge_result : temp_rr[i] + avge_result]
#         # ]
#         print(len(temp_list.tolist()), temp_rr[i])
#         print(temp_list.tolist()[temp_rr[i]])
