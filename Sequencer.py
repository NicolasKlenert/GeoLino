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

    def hasRank(A, k, tol = 0.01):
        if A.size == 0:
            return k == 0
        if k > max(A.shape):
            return False
        return np.linalg.matrix_rank(A, tol) == k

    def doubleDescriptionMethod(A, tol = 0.01, normalize = False, minimise = True):
        m, n = A.shape
        #get a fullrank indexlist
        I = Sequencer._getMinimunIndexList(A,tol)
        #important: V is different here then in the skript! it is transposed
        #therefore V is a Vector of the Vi
        V = np.negative(np.linalg.inv(A[I,:])).T
        J = set(range(m)).difference(I)
        while len(J) != 0:
            j = J.pop()
            a = A[j,:]  #j-th row
            #calcualte scalarproduct of a and vi (vi the columns of V)
            scalars = np.dot(V,a)
            V1 = V[scalars <= 0,:] #all vi with si <= 0
            #create V2: all si*vj-sj*vi with si > 0, sj < 0
            Vneg = V[scalars < 0,:]
            Vpos = V[scalars > 0,:]
            scalars_neg = scalars[scalars < 0]
            scalars_pos = scalars[scalars > 0]
            #calculate rightSummand (sj*vi)
            #at the end we want to slice the tensor (on the third axis; axis=0)
            rightTensor = scalars_neg.reshape((len(scalars_neg),1,1))*Vpos
            rightSummand = rightTensor.reshape((-1,n))
            #calculate leftSummand (si*vj)
            Vneg_m, Vneg_n = Vneg.shape
            leftTensor = scalars_pos.reshape((1,len(scalars_pos),1))*Vneg.reshape((Vneg_m,1,Vneg_n))
            leftSummand = leftTensor.reshape((-1,n))
            V2 = leftSummand - rightSummand
            if minimise:
            #-----delete all unnecessarty Vi------------
                Wneg = np.dot(Vneg, A.T)
                Wpos = np.dot(Vpos, A.T)
                Wpos = np.repeat(Wpos, len(scalars_neg), axis=0)
                Wneg = np.tile(Wneg, (len(scalars_pos),1))
                cond = np.logical_and(Wpos == 0, Wneg == 0)
                #cond array is a boolean matrix corresponding to V2
                indices = [np.nonzero(row)[0] for row in cond]
                #test the rank of AJ
                booleanList = np.array([Sequencer.hasRank(A[ind,:], n-2, tol) for ind in indices])
                V2 = V2[booleanList]
            #------end of extracting unnecessary vi-----------
            V = np.concatenate((V1,V2))
        return V.T
