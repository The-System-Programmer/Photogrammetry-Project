import numpy as np
import os

def Convert_OpenCV_to_Numpy(kp,des,f):
    # Convert keypoints to float 32 and store in a array each
    kp_array = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]for k in kp],dtype=np.float32,)

    # Saves the converted keypoints and descriptors in a .npz for each image
    np.savez(f"Data/Features/{f}.npz", keypoints=kp_array, descriptors=des)
