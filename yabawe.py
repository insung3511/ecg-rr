import wfdb
import numpy as np

DEFAULT_PATH = "./data/mit/100"
anno = wfdb.rdann(DEFAULT_PATH, 'atr')
anno = np.unique(anno.sample[np.in1d(anno.symbol, ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?'])])
print(anno)
