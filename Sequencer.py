import numpy as np
import scipy.linalg as spl

class Sequencer(object):

	def __init__(self, textfile):
		#read textfile out and populate the main variables
		self.M = Sequencer._readTextFile(textfile)
		self.A, self.b = Sequencer.getInput(self.M)

	def run(self):
		self.k, self.U = Sequencer.linSpace(self.A)
		self.M = Sequencer.buildM(self.A, self.U, self.b)
		self.n = self.A.shape[1]
		self.Phat = Sequencer.doubleDescriptionMethod(self.M)
		self.V, self.W = Sequencer.useDDM(self.Phat ,self.n, self.k)
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
	def normalize(A, axis=0, round='toZero', tol = 0.001):
		if round == 'toZero':
			A[np.abs(A) < tol] = 0
		#get the min of every row (by axis=1)
		#print(A)
		mask = np.ma.masked_equal(np.abs(A),0.0,copy=False)
		minimas = np.min(mask, axis=axis)
		#print(minimas)
		scalar = np.divide(np.ones_like(minimas,dtype=float),minimas, out=np.zeros_like(minimas,dtype=float), where= minimas!=0)
		#print(scalar)
		if axis == 1:
			scalar = scalar.reshape((-1, 1))
		elif axis == 0:
			scalar = scalar.reshape((1, -1))
		tmp = np.repeat(scalar,A.shape[axis],axis=axis)
		#print(tmp)
		result = np.multiply(tmp,A)
		if round == 'toZero':
			result[np.abs(result) < tol] = 0
		return result

	@staticmethod
	def doubleDescriptionMethod(A, tol = 0.001, minimise = True, I = False):
		m, n = A.shape
		#get a fullrank indexlist
		if not I:
			I = Sequencer._getMinimunIndexList(A,tol)
		#important: V is different here then in the skript! it is transposed
		#therefore V is a Vector of the Vi
		V = np.negative(np.linalg.inv(A[I,:])).T

		if minimise:
			#round before and after normalizing
			V = Sequencer.normalize(V, axis=1, round='toZero', tol = tol)

		J = set(range(m)).difference(I)
		while len(J) != 0:
			#print(V.T)
			#print("=========================="+str(len(J))+"================")
			j = J.pop()
			a = A[j,:]  #j-th row
			#calcualte scalarproduct of a and vi (vi the columns of V)
			scalars = np.dot(V,a)
			scalars[np.abs(scalars) < tol] = 0
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
			#round small enough numbers to null! -> otherwise we throw to many columns away
			V2[np.abs(V2) < tol] = 0
			if minimise and V2.size > 0:
			#-----delete all unnecessarty Vi------------
				Wneg = np.dot(Vneg, A.T)
				Wpos = np.dot(Vpos, A.T)
				Wneg[np.abs(Wneg) < tol] = 0
				Wpos[np.abs(Wpos) < tol] = 0
				Wneg = np.repeat(Wneg, len(scalars_pos), axis=0)
				Wpos = np.tile(Wpos, (len(scalars_neg),1))

				cond = np.logical_and(Wpos == 0, Wneg == 0)
				#cond array is a boolean matrix corresponding to V2
				indices = [np.nonzero(row)[0] for row in cond]
				#test the rank of AJ
				#d = np.linalg.matrix_rank(A[I,:], tol)
				booleanList = np.array([Sequencer.hasRank(A[ind,:], n-2, tol) for ind in indices])
				V2 = V2[booleanList]
				#delete all vectors we already have
				V2 = Sequencer.normalize(V2, axis=1, round='toZero', tol = tol)
				U = np.tile(V,(V2.shape[0],1,1))
				U2 = V2.reshape(V2.shape[0],1,V2.shape[1])
				uniqueList = np.equal(U,U2).all(axis=2).any(axis=1)
				V2 = V2[np.invert(uniqueList)]
			#------end of extracting unnecessary vi-----------
			V = np.concatenate((V1,V2))
			I.append(j)
		#print(V.T)
		#print("=========================="+str(len(J))+"================")
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
	def linSpace(A):
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
		k = L.shape[1]
		return k, U

	@staticmethod
	def buildM(A, U, b):
		AU = np.dot(A,U)
		up = np.hstack((AU, -b))
		z = np.zeros(AU.shape[1])
		last = np.hstack((z,[-1]))
		M = np.vstack((up,last))
		return M

	@staticmethod
	def useDDM(Phat,n,k):
		# P hat as in 4.11 is the positive hull of the rows of Phat
		Pm, Pn = Phat.shape
		lrow = Phat[Pm-1]
		#print lrow.shape
		nonzero = lrow != 0
		#print nonzero.shape
		scale = 1 / lrow[nonzero]
		#print scale.shape
		#print(Phat)
		nonzero = np.reshape(nonzero, (1,-1))
		nonzero_matrix = np.repeat(nonzero, Pm, axis=0)
		tmp = np.reshape(Phat[nonzero_matrix],(-1,np.sum(nonzero)))
		#print tmp
		V = np.multiply(tmp, scale)
		zero_matrix = np.ones(nonzero_matrix.shape, "bool") - nonzero_matrix
		W = np.reshape(Phat[zero_matrix], (Pm, -1))
		#print(V)
		#print(W)
		#V = V[]
		V = np.delete(V, n-k, 0)
		W = np.delete(W,n-k, 0)
		return V, W

	@staticmethod
	def getInput(M):
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
