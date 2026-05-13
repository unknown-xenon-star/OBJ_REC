# import OpenCV and pyplot 
import cv2 as cv
from matplotlib import pyplot as plt

# read left and right images
imgR = cv.imread('5.png', 0)
imgL = cv.imread('6.png', 0)

# creates StereoBm object 
stereo = cv.StereoBM_create(numDisparities = 16,
                            blockSize = 15)

# computes disparity
disparity = stereo.compute(imgL, imgR)

# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()