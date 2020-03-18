import numpy as np
from  numpy import absolute as cabs
from numpy.linalg import norm
from numpy.linalg import inv
from numpy.linalg import eigvals

#A = np.array([[2,8,3,1],[0,2,-1,4],[7,-2,1,2],[-1,0,5,2]])
A = np.array([[2,-1,0],[-1,3,-1],[0,-1,2]])
#A = np.array([[7,-2,1,2],[2,8,3,1],[-1,0,5,2],[0,2,-1,4]])

Q_gauss = np.zeros(A.shape)
Q_jacobi = np.zeros(A.shape)
I = np.eye(A.shape[0])

for i in range(A.shape[0]) :
	Q_jacobi[i,i] = A[i,i] 
	for j in range(A.shape[0]) :
		if (i >= j ) :
			Q_gauss[i,j] = A[i,j]

B_jacobi = (I-inv(Q_jacobi).dot(A))
B_gauss  = (I-inv(Q_gauss ).dot(A))

Jacobi_radius = cabs(max(eigvals(B_jacobi)))
Gauss_radius = cabs(max(eigvals(B_gauss)))

norm_j = norm(B_jacobi,ord = 2)
norm_g = norm(B_gauss,ord = 2)

print("")
print("Gaussian Radius")
print(Gauss_radius)
print("")
print("Jacobi Radius")
print(Jacobi_radius) 

