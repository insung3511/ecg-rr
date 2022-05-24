from ast import AnnAssign
from re import L
import matplotlib.pyplot as plt
import wfdb.processing as wp
import wfdb

PATH = 'data/nstdb/'
exclude_record = ['bw', 'em', 'ma']
record_list = []

print("[INFO] Read records file from ", PATH)
with open(PATH + 'RECORDS') as f:
    record_lines = f.readlines()

# Read Records
for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))
for i in range(len(record_lines)):
    record_lines[i] = record_lines[i].strip()

print(record_list)

for i in range(len(record_lines)):
    r_path = PATH + record_lines[i]
    sig, fields = wfdb.rdsamp(r_path)
    rrin = wp.ann2rr(r_path, 'atr', as_array=True, start_time=0.0)
    xqrs = wp.XQRS(sig=sig[:, 0], fs=fields['fs'])
    
    xqrs.detect()
    print(sig, rrin)

pre_rrin_temp = 0
post_rrin_temp = 0

# for i in range(len(sig)):
#     post_rrin_temp += rrin[i]

#     plt.plot(sig[pre_rrin_temp : post_rrin_temp])
#     plt.title(i)
#     plt.show()

#     pre_rrin_temp += post_rrin_temp
#     print(pre_rrin_temp, post_rrin_temp, rrin[i])