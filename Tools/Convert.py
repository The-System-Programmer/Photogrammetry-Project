# Import necessary libraries
import cv2
import os
import numpy as np

results = []
def convert():
    # List containing name of all the .npz files
    data_dir = "Data/Features"
    files = sorted(os.listdir(data_dir))

    # Iterate through each .npz files and convert them back to opencv object
    for f in files:
        data = np.load(f"Data/Features/{f}")
        kp_array =data['keypoints']
        descriptors = data['descriptors']

        keypoints = []
        for row in kp_array:
            x,y,size,angle,response,octave,class_id = row
            kp = cv2.KeyPoint(x,y,size,angle,response,int(octave),int(class_id)) # type: ignore
            keypoints.append(kp)
        results.append((keypoints,descriptors))
    return results
