import wfdb.processing as wp
import wfdb

R_PATH = "./data/mit/"

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

# R-R Interval
for i in range(len(record_list)):
    temp_avge_cnt = 0
    target_list = []
    temp_path = R_PATH + record_list[i]
    temp_rr = (wp.ann2rr(temp_path, 'atr', as_array=True, start_time=0.0))
    temp_cnt = -len(temp_rr)

    for i in range(len(temp_rr)):
        avge_result = (temp_rr[temp_cnt] + temp_rr[(temp_cnt + 1)]) / 2     # Might be window size
        record_avge.append(avge_result)                                     # Save it...
        temp_cnt += 1                       
            
        if longest <= avge_result:                                          # longest windows size selecting
            longest = avge_result

print("Longest strength : {},\tLongest Average : {}".format(longest, longest / 2))

for i in range(len(record_list)):
    temp_path = R_PATH + record_list[i]
    temp_list = wfdb.rdsamp(temp_path, sampfrom=0)

    temp_rr = (wp.ann2rr(temp_path, 'atr', as_array=True, start_time=0.0))
    
    for i in range(len(temp_list)):
        zero_padded_list = [
            temp_list[temp_rr[i] - avge_result[i] : temp_rr[i] + avge_result[i]]
        ]
        print(zero_padded_list)