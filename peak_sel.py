from os import pread
import matplotlib.pyplot as plt
import wfdb

R_PATH = "./data/nstdb/"
exclude_record = ["bw", "em", "ma"]

zero_padded_list = []
record_list = []

record_ann = []

# Read RECORDS txt file
print("[INFO] Read records file from ", R_PATH)
with open(R_PATH + 'RECORDS') as f:
    record_lines = f.readlines()

# Read Records
for i in range(len(record_lines)):
    record_list.append(str(record_lines[i].strip()))

print('''{}
{}'''.format(R_PATH, record_list))

for i in range(len(record_list)):
    temp_rpath = R_PATH + record_list[i]

    # Read original sample by rdsamp function
    record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[1], sampfrom=0)

    # Got R-R Peak by rdann funciton
    record_ann = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).sample)[1:]

    for i in range(len(record_ann)):
        pre_add = record_ann[i - 1]
        post_add = record_ann[i + 1]
        print(pre_add, post_add)

        if i < 1:
            pre_add = 0

        plt.title(temp_rpath)
        plt.grid(color='0.5')
        plt.plot(record_sg[pre_add:post_add])
        plt.show()