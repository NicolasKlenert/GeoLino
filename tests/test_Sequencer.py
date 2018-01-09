from .. import Sequencer as Se
import numpy as np

def test_matrix_construction():
    filepath = 'tests/text.txt'
    matrix = Se.Sequencer._readTextFile(filepath)
    assert np.array_equal(matrix, [[ 1.  ,  0.  ,  1.  ,  0.  ,  1.  ],[ 0.5 ,  4.  ,  3.  ,  2.  ,  1.  ],[-1.  , -0.25,  4.  ,  2.  ,  0.  ]]) == True

def test_indexlist():
    assert Se.Sequencer._getMinimunIndexList(np.eye(4)) == [0,1,2,3]
    arr = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
    assert Se.Sequencer._getMinimunIndexList(arr) == [0,1,3]

def test_doubleDescriptionMethod():
    A1 = np.negative(np.eye(3))
    A2 = np.array([[1,-1,-1],[1,-2,1]])
    A = np.concatenate((A1,A2),axis=0)
    assert np.array_equal(Se.Sequencer.doubleDescriptionMethod(A,minimise=False),[[0,1,0,2,1,3],[1,1,1,2,1,2],[0,0,2,2,1,1]])
    assert np.array_equal(Se.Sequencer.doubleDescriptionMethod(A,minimise=True),[[0,1,0,3],[1,1,1,2],[0,0,2,1]])

def test_Sequencer():
    filepath = 'tests/cube.poly'
    seq = Se.Sequencer(filepath)
    V, W = seq.run()
    assert W.size == 0
    assert np.array_equal(V,[[]])
