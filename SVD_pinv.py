import numpy as np
from numpy.linalg import svd
from numpy.linalg import pinv

A = np.array([[2,3],[3,4],[2,8]])
(m,n) = A.shape
U, s, V_trans = svd(A)
V = np.transpose(V_trans)
U_trans = np.transpose(U)
s_inv = 1/s
Sigma_inv = np.zeros((n,m))
Sigma = np.zeros(A.shape)
Sigma_inv[:2,:2] = np.diag(s_inv)
Sigma[:2,:2] = np.diag(s)


A_pinv = V@Sigma_inv@U_trans

print(A_pinv)
print("")

print(Sigma@Sigma_inv)
print("")

print(Sigma_inv@Sigma)
print("")