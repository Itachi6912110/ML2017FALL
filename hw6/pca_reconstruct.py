from skimage.io import ImageCollection, imsave, imshow, imread
from numpy import transpose as T
import numpy as np
import sys

#################################################
#                  Functions                    #
#################################################
def flatten_img(img):
	img0 = img[:,:,0].reshape((1,-1))
	img1 = img[:,:,1].reshape((1,-1))
	img2 = img[:,:,2].reshape((1,-1))
	img_new = np.concatenate((img0,img1),axis=1)
	img_new = np.concatenate((img_new,img2),axis=1)

	return img_new

def toimg(img):
	img = img * 1.0
	img -= np.min(img)
	img /= np.max(img)
	img = (img * 255).astype(np.uint8)
	return img

def list2img(img):
	h = 600
	w = 600
	img = img.reshape((h*w*3,))
	r = img[0:h*w].reshape(h,w,1)
	g = img[h*w:h*w*2].reshape(h,w,1)
	b = img[h*w*2:h*w*3].reshape(h,w,1)
	new_img = np.concatenate((r,g),axis=2)
	new_img = np.concatenate((new_img,b),axis=2)
	new_img = toimg(new_img)
	return new_img

#################################################
#                    Main                       #
#################################################
#doing reconstruction
img = imread(sys.argv[1]+"/"+sys.argv[2])
img = flatten_img(img)
U = np.load("eigfaces_4.npy")
mean_face = np.load("mean_face.npy")
img = img - mean_face
re_img = np.dot(T(np.dot(U, T(img))),U) + mean_face
re_img = list2img(re_img)
imsave('reconstruction.jpg', re_img)