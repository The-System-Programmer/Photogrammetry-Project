import numpy as np
import os

def keypoints_and_des(kp,des,f):
    # Convert keypoints to float 32 and store in a array each
    kp_array = np.array([[k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave, k.class_id]for k in kp],dtype=np.float32,)
    return kp_array
