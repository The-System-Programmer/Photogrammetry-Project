import numpy as np
import cv2

from Pipeline.Step2_Feature_matching import feature_matching
from Tools.Convert_Numpy_to_OpenCV import Convert_Numpy_to_OpenCV


def compute_camera_geometry():
    matches_all = feature_matching()
    data = Convert_Numpy_to_OpenCV()

    camera_pairs = []

    for entry in matches_all:
        if entry["num_matches"] < 8:
            continue

        i, j = entry["pair"]
        matches = entry["matches"]

        kp1, des1 = data[i]
        kp2, des2 = data[j]

        # --- Extract matched keypoint coordinates ---
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # --- Fundamental Matrix ---
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=1.0,
            confidence=0.99
        )

        if F is None:
            continue

        # --- Filter inliers ---
        inliers = mask.ravel() == 1
        pts1 = pts1[inliers]
        pts2 = pts2[inliers]

        if len(pts1) < 8:
            continue

        # --- Camera 1 (canonical) ---
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))

        # --- Compute epipole ---
        U, S, Vt = np.linalg.svd(F.T)
        e = Vt[-1]

        # Avoid division crash
        if abs(e[2]) < 1e-8:
            continue

        e = e / e[2]

        # --- Skew-symmetric matrix ---
        e_x = np.array([
            [0, -e[2], e[1]],
            [e[2], 0, -e[0]],
            [-e[1], e[0], 0]
        ])

        # --- Camera 2 ---
        P2 = np.hstack((e_x @ F, e.reshape(3, 1)))

        camera_pairs.append({
            "pair": (i, j),
            "P1": P1,
            "P2": P2,
            "F": F,
            "pts1": pts1,
            "pts2": pts2,
            "num_inliers": len(pts1)
        })

    return camera_pairs
