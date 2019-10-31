import numpy as np
from numpy.linalg import svd
from numpy.linalg import norm

A = np.array([[2,3],[3,4],[2,8]])
(m,n) = A.shape
U, s, V_trans = svd(A)
Sigma = np.zeros(A.shape)
Sigma[:n,:n] = np.diag(s)

A_constructed = U.dot(Sigma).dot(V_trans)
print("Original Matrix :")
print(A)
print("")
print("Constructed Matrix : ")
print(A_constructed)
print("")
print("Norm of the Difference ", norm(A - A_constructed))