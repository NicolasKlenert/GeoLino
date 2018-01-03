import Sequencer as Se
import numpy as np

def test_matrix_construction():
    filepath = 'tests/text.txt'
    matrix = Se.Sequencer._readTextFile(filepath)
    assert np.array_equal(matrix, [[ 1.  ,  0.  ,  1.  ,  0.  ,  1.  ],[ 0.5 ,  4.  ,  3.  ,  2.  ,  1.  ],[-1.  , -0.25,  4.  ,  2.  ,  0.  ]]) == True

def test_indexlist():
    assert Se.Sequencer._getMinimunIndexList(np.eye(4)) == [0,1,2,3]
    arr = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
    assert Se.Sequencer._getMinimunIndexList(arr) == [0,1,3]
