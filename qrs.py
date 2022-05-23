from ast import AnnAssign
import matplotlib.pyplot as plt
import wfdb.processing as wp
import wfdb

sig, fields = wfdb.rdsamp('data/mit/100')
rrin = wp.ann2rr('data/mit/100', 'atr', as_array=True, start_time=0.0)
xqrs = wp.XQRS(sig=sig[:, 0], fs=fields['fs'])
xqrs.detect()

pre_rrin_temp = 0
post_rrin_temp = 0
for i in range(len(sig)):
    post_rrin_temp += rrin[i]

    plt.plot(sig[pre_rrin_temp : post_rrin_temp])
    plt.title(i)
    plt.show()

    pre_rrin_temp += post_rrin_temp
    print(pre_rrin_temp, post_rrin_temp, rrin[i])