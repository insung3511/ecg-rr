import matplotlib.pyplot as plt
import wfdb.processing as wp
import wfdb

R_PATH = "./data/nstdb/"
exclude_record = ["bw", "em", "ma"]

zero_padded_list = []
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

# Find Longest R-R Interval for window
for i in range(len(record_list)):
    temp_rpath = R_PATH + record_list[i]
    interval = wp.ann2rr(temp_rpath, 'atr', as_array=True)
    longest = interval.max()
print("LONGEST: ", longest)

for i in range(len(record_list)):
    temp_rpath = R_PATH + record_list[i]

    # Read original sample by rdsamp function
    record_sg, _ = wfdb.rdsamp(temp_rpath, channels=[0], sampfrom=0)

    # Got R-R Peak by rdann funciton
    record_ann = list(wfdb.rdann(temp_rpath, 'atr', sampfrom=0).sample)[1:]

    for i in range(len(record_ann)):
        pre_add = record_ann[i - 1]
        post_add = record_ann[i + 1]

        cut_pre_add = record_ann[i] - int((record_ann[i] - pre_add) / 2) 
        cut_post_add = record_ann[i] + int((post_add - record_ann[i]) / 2)

        print(pre_add, post_add, "\t\t\t", cut_pre_add, cut_post_add)

        if i < 1:
            pre_add = 0
        
        windowed_list.append(record_sg[pre_add:post_add])

        plt.title(temp_rpath)
        plt.grid(color='0.5')
        plt.plot(record_sg[cut_pre_add:cut_post_add])
        plt.plot(record_sg[pre_add:post_add])
        plt.show()