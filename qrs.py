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
    if record_lines[i] == exclude_record[i]:
        pass
    rpath = PATH + record_lines[i]
    wp.ann2rr(rpath, 'atr', start_time=0)

# for i in range(len(sig)):
#     post_rrin_temp += rrin[i]

#     plt.plot(sig[pre_rrin_temp : post_rrin_temp])
#     plt.title(i)
#     plt.show()

#     pre_rrin_temp += post_rrin_temp
#     print(pre_rrin_temp, post_rrin_temp, rrin[i])