import Sequencer as Se
import numpy as np

filepath = 'tests/homework.poly'
seq = Se.Sequencer(filepath)
V, W = seq.run()
r = np.linalg.matrix_rank(V)
print r
#print np.sum(V == 0) + np.sum(V==1) + np.sum(V==-1)
print V
print W
