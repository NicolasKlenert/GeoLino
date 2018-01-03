import numpy as np

class Sequencer(object):

    def __init__(self, textfile):
        #read textfile out and populate the main variables
        self.matrix = Sequencer._readTextFile(textfile)

    def _readTextFile(url):
        #outputs an numpy matrix
        # the first index describes on which row, the second the columns
        with open(url) as f:
            tupel = [int(string) for string in f.readline().rstrip().split(' ')]
            matrix = np.empty(tupel)
            content = f.readlines()
            for i in range(tupel[0]):
                matrix[i] = [float(num) for num in content[i].rstrip().split(' ')]
        return matrix

    def _getMinimunIndexList(A, tol = 0.01):
        #if rank of A is not n, throw error
        m, n = A.shape
        if np.linalg.matrix_rank(A, tol) != n:
            raise Exception("The rank of the given matrix has to be full")
        #find a subset of [n] so that A_I has still full rank
        #this option (just testing the rank everytime) is really time consuming
        #there are better ways
        I = []
        currentRank = 0
        for i in range(m):
            J = I +[i]
            rank = np.linalg.matrix_rank(A[J,:], tol)
            if rank > currentRank:
                I = J
                currentRank = rank
                if currentRank == n:
                    break
        #I should be the minimal indexlist with full rank
        return I

    def doubleDescriptionMethod(A, tol = 0.01):
        #get a fullrank indexlist
        I = Sequencer._getMinimunIndexList(A,tol)
