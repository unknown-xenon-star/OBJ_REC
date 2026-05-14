D_real = 30.5
fx= 509.5081967213115

import cv2
import numpy as np

# After calibrating your camera:
# ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera()

# fx = camera_matrix[0][0]  # focal length in pixels (x-axis)
# fy = camera_matrix[1][1]  # focal length in pixels (y-axis)

def real_distance(dis_app, fx=fx):
    Z = (D_real * fx) / dis_app
    return Z

def fget(Z=155.4, d=100):
    f = (Z * d) / D_real 
    return f

print(fget())
