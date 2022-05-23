from ast import AnnAssign
import wfdb.processing as wp
import wfdb

sig, fields = wfdb.rdsamp('data/mit/100')
xqrs = wp.XQRS(sig=sig[:, 0], fs=fields['fs'])
xqrs.detect()

print(xqrs.rr_init)