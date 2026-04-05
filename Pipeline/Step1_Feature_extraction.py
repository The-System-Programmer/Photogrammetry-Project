import os
import cv2 as cv

def features():
    results = []
    files = os.listdir("Data/Pictures")
    for f in files:
        print(f"Extracting features and description of {f}")
        path = f"Data/Pictures/{f}"
        img = cv.imread(path, 0)
        orb = cv.ORB_create()  # type: ignore

        # Keypoints
        kp = orb.detect(img, None)

        # Description
        kp,ds = orb.compute(img, kp)
        results.append({"path":path,"kp":kp,"ds":ds})
    return results
