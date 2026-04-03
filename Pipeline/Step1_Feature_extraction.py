import os
import cv2 as cv

def features():
    results = []
    files = os.listdir("Data/Pictures")
    for f in files:
        path = f"Data/Pictures/{f}"
        img = cv.imread(path, 0)
        orb = cv.ORB_create()  # type: ignore

        # Keypoints
        kp = orb.detect(img, None)

        # Description
        kp,ds = orb.compute(img, kp)
        results.append({"name":f,"kp":kp,"ds":ds})
    return results
