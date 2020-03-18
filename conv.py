import numpy as np
import cv2 as cv

def padding(img, pad) :
	img_borders = cv.copyMakeBorder(img, pad, pad, pad, pad, cv.BORDER_CONSTANT)
	return img_borders

def convolution(image, filtr, stride = 1) :
	(m,n) = filtr.shape
	pad = m
	padded = padding(image, pad)
	(r,c) = padded.shape
	o_m = int((r - m + 1)/stride)
	o_n = int((c - n + 1)/stride)
	convolved = np.zeros((o_m,o_n),dtype = "float")
	for i in np.arange(o_m) :
		for j in np.arange(o_n) :
			temp = np.zeros(filtr.shape)
			temp = padded[i*stride : i*stride + m , j*stride : j*stride + n ]*filtr
			convolved[i][j] = temp.sum()
	convolved /= np.amax(convolved)
	return convolved
            
smallBlur = np.ones((7,7) , dtype = "float") * (1.0/(7*7))
largeBlur = np.ones((21,21) , dtype = "float") * (1.0/(21*21))

sharpen = np.array((
	[0 , -1, 0 ],
	[-1, 5 , -1],
	[0, -1 , 0]) , dtype = "int")

laplacian = np.array((
	[0 ,  1 , 0],
	[1 , -4 , 1],
	[0 ,  1 , 0]) ,dtype = "int")

sobelX = np.array((
	[-1 , 0 , 1],
	[-2 , 0 , 2],
	[-1 , 0 , 1]), dtype = "int")

sobelY = np.array((
	[-1 , -2 , -1],
	[0  ,  0 ,  0],
	[1  ,  2 ,  1]) , dtype = "int")

emboss = np.array((
	[-2 , -1, 0],
	[-1 ,  1, 1],
	[ 0 ,  1, 2]) , dtype = "int")

kernelBank = (
	("small_blur" , smallBlur),
	("large_blur" , largeBlur),
	("sharpen"    , sharpen),
	("laplacian"  , laplacian),
	("sobel_x"    , sobelX),
	("sobel_y"    , sobelY),
	("emboss"     , emboss))


img = cv.imread("messi.jpg",0)
output = convolution(image, sharpen, 1)
cv.imshow("Blur",output)
cv.waitKey(0)
cv.destroyAllWindows()