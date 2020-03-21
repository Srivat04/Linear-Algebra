import numpy as np
import cv2 as cv

img = cv.imread("messi.jpg",0)
(m,n) = img.shape
img_borders = cv.copyMakeBorder(img,int(m/15),int(m/15),int(n/15),int(n/15),cv.BORDER_CONSTANT)
cv.imshow("Image",img)
cv.imshow("Imgae_border",img_borders)
cv.waitKey(0)
cv.destroyAllWindows()
