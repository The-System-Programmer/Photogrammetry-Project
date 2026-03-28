# Import necessary libraries
import os
import cv2
import numpy as np

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

    # Convert keypoints to float 32 and store in a array each
    kp_array = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]for k in kp],dtype=np.float32,)

    # Saves the converted keypoints and descriptors in a .npz for each image
    np.savez(f"Data/Features/{f}.npz", keypoints=kp_array, descriptors=des)
