from .. import Sequencer as Se
import numpy as np

def sameProducers(A,B):
    if not A.shape == B.shape:
        return False
    A = Se.Sequencer.normalize(A)
    B = Se.Sequencer.normalize(B)
    if np.allclose(A,B):
        return True
    listA = [row for row in A.T]
    listB = [row for row in B.T]
    for colA in listA:
        for idx,colB in enumerate(listB):
            if np.allclose(colA, colB):
                del listB[idx]
                break
    return not bool(listB)

def test_sameProducers():
    A =  np.arange(1,10).reshape((3,3))
    B = np.array([[1,1,1],[4,2.5,2],[7,4,3]])
    C = np.array([[4,2.5,2],[7,4,3],[1,1,1]])
    D = np.array([[1,1,1],[2.5,2,4],[4,3,7]])
    assert sameProducers(A,B)
    assert not sameProducers(A,C)
    assert sameProducers(A,D)
    E = np.array([[0,1,1,6],[0,1,4,6],[1,1,1,0],[0,1,1,3]])
    F = np.array([[1/3.0,1/3.0,2,0],[4/3.0,1/3.0,2,0],[1/3.0,1/3.0,0,1],[1/3.0,1/3.0,1,0]])
    assert sameProducers(F,E)

def test_matrix_construction():
    filepath = 'tests/text.txt'
    matrix = Se.Sequencer._readTextFile(filepath)
    assert np.array_equal(matrix, [[ 1.  ,  0.  ,  1.  ,  0.  ,  1.  ],[ 0.5 ,  4.  ,  3.  ,  2.  ,  1.  ],[-1.  , -0.25,  4.  ,  2.  ,  0.  ]]) == True

def test_indexlist():
    assert Se.Sequencer._getMinimunIndexList(np.eye(4)) == [0,1,2,3]
    arr = np.array([[1,0,0],[0,1,0],[1,1,0],[0,0,1]])
    assert Se.Sequencer._getMinimunIndexList(arr) == [0,1,3]

def test_normalize_Rows():
    A = np.array([[0,1,3],[0,3,3],[-3,-3,-4]])
    B = np.array([[0,1,3],[0,1,1],[-1,-1,-1.3333333]])
    assert np.allclose(Se.Sequencer.normalize(A,axis=1),B)

def test_DDM_1():
    A1 = np.negative(np.eye(3))
    A2 = np.array([[1,-1,-1],[1,-2,1]])
    A = np.concatenate((A1,A2),axis=0)
    B1 = np.array([[0,1,0,2,1,3],[1,1,1,2,1,2],[0,0,2,2,1,1]])
    B2 = np.array([[0,1,0,3],[1,1,1,2],[0,0,2,1]])
    V1 = Se.Sequencer.doubleDescriptionMethod(A,minimise=False)
    V2 = Se.Sequencer.doubleDescriptionMethod(A,minimise=True)
    assert sameProducers(V1,B1)
    assert sameProducers(V2,B2)
    #assert np.array_equal(Se.Sequencer.doubleDescriptionMethod(A,minimise=False),[[0,1,0,2,1,3],[1,1,1,2,1,2],[0,0,2,2,1,1]])
    #assert np.array_equal(Se.Sequencer.doubleDescriptionMethod(A,minimise=True),[[0,1,0,3],[1,1,1,2],[0,0,2,1]])

def test_DDM_2():
    A = np.array([[1,1,1,1],[2,1,1,1],[1,2,1,1],[2,2,3,1],[1,1,5,1]])
    B1 = np.array([[1,0,-1,2,-2,0],[1,0,2,-1,0,-2],[-0.5,-0.5,0.5,0.5,0,0],[-2.5,0.5,-3.5,-3.5,2,2]])
    B2 = np.array([])#look at B2
    V1 = Se.Sequencer.doubleDescriptionMethod(A,minimise= False)
    V2 = Se.Sequencer.doubleDescriptionMethod(A,minimise= True)
    #print(V1)
    assert sameProducers(V1,B1) #the arrays are not 100% equal because of floit point aritmetic

def test_DDM_3():
    A = np.array([[-1,0,0,1],[-1,0,-1,2],[0,-1,0,1],[0,-1,-1,2],[1,1,-1,-4],[1,1,0,-5],[0,0,0,-1]])
    B1 = np.array([[0,1,1,4,6,1,4,5,6,9,15],[0,1,4,1,6,4,1,5,9,6,15],[1,1,1,1,0,4,4,8,3,3,6],[0,1,1,1,3,1,1,2,3,3,6]])
    B2 = np.array([[0,1,1,4,2],[0,1,4,1,2],[1,1,1,1,0],[0,1,1,1,1]])
    V1 = Se.Sequencer.doubleDescriptionMethod(A,minimise= False, I=[0,1,2,6])
    #the thing is, that we get a whole different solution with I=[0,1,2,4] (normal start index)
    V2 = Se.Sequencer.doubleDescriptionMethod(A,minimise=True, I=[0,1,2,6])
    V3 = Se.Sequencer.doubleDescriptionMethod(A,minimise=True)
    assert np.array_equal(V1,B1) #the arrays are not 100% equal because of floit point aritmetic
    assert sameProducers(V2,V3)
    assert sameProducers(V2,B2) #-> same Producers no matter of start index, if minimise is on!

def test_Sequencer_Cube():
    filepath = 'tests/cube.poly'
    seq = Se.Sequencer(filepath)
    V, W = seq.run()
    assert W.size == 0
    assert np.array_equal(V,[[1,-1,1,-1,1,-1,1,-1],[-1,-1,1,1,-1,-1,1,1],[-1,-1,-1,-1,1,1,1,1]])

def test_Sequencer_Homework():
    pass
    #filepath = 'tests/homework.poly'
    #seq = Se.Sequencer(filepath)
    #V, W = seq.run()
    #assert np.array_equal(seq.M, [])
    #assert sameProducers(V,np.array([[0,2],[1,0],[0,0]]))
    #assert sameProducers(W,np.array([[1,15,0,4,0,2],[7,0,2,0,1,0],[5,5,1,1,0,0]]))

def test_Sequencer_Crosspolytope():
    seq = Se.Sequencer("tests/crosspolytope.poly")
    V,W = seq.run()
    B1 = np.eye(4)
    B2 = -np.eye(4)
    B = np.concatenate((B1,B2), axis=1)
    assert W.size == 0
    assert sameProducers(V, B)

def test_Sequencer_Cubactahedron():
    seq = Se.Sequencer("tests/cuboctahedron.poly")
    V,W = seq.run()
    assert W.size == 0
    assert V.shape == (3,12)
