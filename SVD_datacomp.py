import argparse
import numpy as np
from numpy.linalg import svd
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument("-c","--compfactor", help = "Factor by which the image is compressed", default = 4, type = int )
args = parser.parse_args()
c_fact = args.compfactor

img = cv.imread("messi.jpg")

A = img[:,:,0]
U , s , V_trans = svd(A)

Sigma = np.zeros(A.shape)
Sigma[:min(A.shape),:min(A.shape)] = np.diag(s)

B = np.zeros(A.shape)

for i in range(int(len(s)/c_fact)) :

    u = np.zeros(U.shape)
    v_t = np.zeros(V_trans.shape)
    sig = np.zeros(Sigma.shape)

    u[:,i] = U[:,i]
    v_t[i,:] = V_trans[i,:]
    sig[i,i] = Sigma[i,i]    

    B += u@sig@v_t
#cv.imshow("original_image",img[:,:,0])
#cv.imshow("compressed_image",B)
#cv.waitKey(0)
#cv.destroyAllWindows()
B /= np.amax(B) 

cv.imshow("original_image",img[:,:,0])
cv.imshow("compressed_image",B)
cv.waitKey(0)
cv.destroyAllWindows()