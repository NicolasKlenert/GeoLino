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

    # def _calculateTensor(A,s, repeatFirstScalar = True):
    #     Am,An = A.shape
    #     sn = len(s)
    #     if repeatFirstScalar:
    #         A = A.reshape((1,Am,An)) #add third dimension
    #         s = s.reshape((sn,1,1)) #point vector to third axis (in python axis=0)
    #         #we reduce the third axis of A (so axis=0) and the first axis of s (axis = 2)
    #         axis = ([2],[0])
    #     else: #makes no difference!
    #         A = A.reshape((Am,1,An))
    #         s= s.reshape((1,sn,1))
    #         #just take the index of a 1
    #         axis = ([2],[1])
    #     #now multiply so we get an 3dim tensor
    #     return np.tensordot(s,A,axes=axis)
    #     #for repeating of tensors: look at np.tile and np.repeat
    #     ALL unnecessary! Tensordot automatacly creates new dimensions if needed

    def doubleDescriptionMethod(A, tol = 0.01):
        m, n = A.shape
        #get a fullrank indexlist
        I = Sequencer._getMinimunIndexList(A,tol)
        #important: V is different here then in the skript! it is transposed
        #therefore V is a Vector of the Vi
        V = np.negative(np.linalg.inv(A[I,:])).T
        J = set(range(n)).difference_update(I)
        while len(J) != 0:
            j = J.pop()
            a = A[j,:]  #j-th row
            #calcualte scalarproduct of a and vi (vi the columns of V)
            scalars = np.dot(V,a)
            #TODO: delete all unnecessarty Vi
            V1 = V[scalars <= 0] #all vi with si <= 0
            #create V2: all si*vj-sj*vi with si > 0, sj < 0
            Vneg = V[scalars < 0]
            Vpos = V[scalars > 0]
            scalars_neg = scalars[scalars < 0]
            scalars_pos = scalars[scalars > 0]
            #calculate rightSummand (sj*vi)
            #at the end we want to slice the tensor (on the third axis; axis=0)
            rightSummand = np.tensordot(Vpos,scalar_neg, axes=0).reshape((-1,v_n),order='F')
            #calculate leftSummand (si*vj)
            leftSummand = np.tensordot(Vneg, scalar_pos, axes= 0).reshape((-1,v_n),order='C')
            V2 = leftSummand - rightSummand
            #TODO: delete all unnecessarty Vi
            V = numpy.concatenate((V1,V2))
        return V
