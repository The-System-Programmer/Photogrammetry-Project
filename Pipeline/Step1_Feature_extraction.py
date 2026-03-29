# Import necessary libraries
import os
import cv2
import numpy as np
from Tools.Convert_OpenCV_to_Numpy import Convert_OpenCV_to_Numpy
# List containg name of all the images
data_dir = "Data/Pictures"
files = sorted(os.listdir(data_dir))

# Make folder
if not os.path.exists("Data/Features"):
    os.makedirs("Data/Features")

# Create a ORB object
orb = cv2.ORB_create()

# Iterate through images using list
for f in files:
    # Loads the image converts to gray scale
    img = cv2.imread(f"Data/Pictures/{f}", 0)

    # Computes the keypoints and descriptions of the image
    kp, des = orb.detectAndCompute(img, None)
    Convert_OpenCV_to_Numpy(kp,des,f)
