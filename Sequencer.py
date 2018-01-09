import numpy as np
import scipy.linalg as spl

class Sequencer(object):

    def __init__(self, textfile):
        #read textfile out and populate the main variables
        self.M = Sequencer._readTextFile(textfile)
        self.A, self.b = Sequencer._get_input(self.M)

    def run(self):
        self.L, self.U = Sequencer._lin_space(self.A)
        self.V, self.W = Sequencer._use_DDM(self.A, self.L, self.U, self.b)
        return (self.V, self.W)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def hasRank(A, k, tol = 0.01):
        if A.size == 0:
            return k == 0
        if k > max(A.shape):
            return False
        return np.linalg.matrix_rank(A, tol) == k
        
    @staticmethod
    def doubleDescriptionMethod(A, tol = 0.01, minimise = True):
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
            if minimise and V2.size > 0:
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

    @staticmethod
    def nullspace(A, atol=1e-13, rtol=0):
        """Compute an approximate basis for the nullspace of A.
    
        The algorithm used by this function is based on the singular value
        decomposition of `A`.
    
        Parameters
        ----------
        A : ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
        atol : float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
        rtol : float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.
    
        If both `atol` and `rtol` are positive, the combined tolerance is the
        maximum of the two; that is::
            tol = max(atol, rtol * smax)
        Singular values smaller than `tol` are considered to be zero.
    
        Return value
        ------------
        ns : ndarray
            If `A` is an array with shape (m, k), then `ns` will be an array
            with shape (k, n), where n is the estimated dimension of the
            nullspace of `A`.  The columns of `ns` are a basis for the
            nullspace; each element in numpy.dot(A, ns) will be approximately
            zero.
        """
    
        A = np.atleast_2d(A)
        u, s, vh = np.linalg.svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

    @staticmethod
    def _lin_space(A):
    	# compute linearity space L of the polyhedron P={x in R^n: Ax<=b}
    	# L be s.t. for an ONB u_1, ..., u_k, ..., u_n of R^n L = lin{u_1, ..., u_k} and
    	# U = [u_k+1, ..., u_n] with lin{u_k+1, ..., u_n} the orth. complement of L
    	# be a_1, ..., a_m the rows of A, then the orth. complement of L is lin{a_1, ..., a_m}
    	# TODO: suitable error tolerance, use floating point arithmetic
    	m, n = A.shape
    	L = Sequencer.nullspace(A, atol=1e-13, rtol=0)
    	U = np.eye(n)
    	# ONB only exists if L not zero, in case that L is zero nullspace(...) returns []
    	if L.size > 0:
    		L = spl.orth(L)
    		U = spl.orth(np.transpose(A))
    	else:
    		L = np.zeros((1,1))
    	return (L, U)

    @staticmethod
    def _use_DDM(A, L, U, b):
        # ...
        m, n = A.shape
        print(L)
        if np.array_equal(L,[[0]]):
            k = 0
        else:
            k = L.shape[1] # dimension of linearity space L
        AU = np.dot(A,U)
        up = np.hstack((AU, -b))
        z = np.zeros(AU.shape[1])
        last = np.hstack((z,[-1]))
        M = np.vstack((up,last))
        # P head as in 4.11 is the positive hull of the rows of Phead
        Phead = Sequencer.doubleDescriptionMethod(M)
        # note that DoubleDescriptionMethod delivered in the past a Matrix where the ROWS the desired vecctors
        Phead = Phead.T
        numbvecs = Phead.shape[0]
        # V and W will contain the desired vectors as columns
        V = []
        W = []
        i = 0
        print n
        print k
        print Phead
        elem = Phead[i][n-k]
        # scale rows s.t. last elements are 1 (or 0 per construction)
        print elem
        while elem != 0:
    		v = Phead[i]
    		if elem != 1:
    			v = v / elem
    		# get rid of last element which is 1
    		v = np.delete(v, n-k+1)
    		if i == 0:
    			V = v
    		else:
    			V = np.vstack((V, v))
    		i = i + 1
    		  # in case that all last elements are 0, the polyhedron is empty
        if i != 0:
          for j in range (i, numbvecs):
              w = Phead[j]
              w = np.delete(w, n-k+1)
              if j == i:
                  W = w
              else:
                 W = np.vstack((W, w))
    	return V.T, W.T

    @staticmethod
    def _get_input(M):
    	# not very smart, better do that when reading file
    	# splits the matrix M (which was read from the text file) into A and b
    	# b is needes as a column vector
    	m, l = M.shape
    	n = l - 1
    	# create b
    	b = np.empty((m, 1))
    	for i in range (0, m):
    		b[i] = M[i][l-1]
    	# create A
    	A = np.empty((m, n))
    	for i in range (0, m):
    		for j in range (0, n):
    			A[i][j] = M[i][j]
    	return A, b
