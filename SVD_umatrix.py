import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm
from numpy.linalg import eig
from numpy.linalg import det
from numpy.linalg import inv

M = np.array([[2,3],[3,4],[2,8]])

A1 = M.dot(np.transpose(M))
A2 = np.transpose(M).dot(M)

U, sigma, V_trans = svd(M)
V = np.transpose(V_trans)

e1,eigenvectors_aat = eig(A1)
e2,eigenvectors_ata = eig(A2)

for i in range(3):
	U[:][i] = U[:][i]/norm(U[:][i])

for i in range(2):
	V[:][i] = V[:][i]/norm(V[:][i])


print("")
print("Matrix U with normalized columns")
print(U)
print("")
print("Matrix with normalized eigenvectors") #The U matrix is the eigenvectors of A*A^t arranged in columns
print(eigenvectors_aat)


print("")
print("Matrix V with normalized columns")
print(V)									#The V matrix is the eigenvectors of A^t*A arranged in columns
print("")
print("Matrix with normalized eigenvectors")
print(eigenvectors_ata)

#The M matrix can be reconstructed from U,V and their eigenvalues by the following equation M = U*D*V^t