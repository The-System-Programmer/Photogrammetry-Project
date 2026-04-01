import cv2
from Pipeline.Step1_Feature_extraction import feature_extraction
def feature_matching(ratio_thresh = 0.75,min_matches=10):
    data = feature_extraction()
    matches_all = []

    #ORB using hamming distance
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck = False)

    for i in range(len(data)):
        kp1,des1 = data[i]

        for j in range(i+1,len(data)):
            kp2,des2 = data[j]

            #KNN matches
            matches = bf.knnMatch(des1,des2,k=2)

            # Lows ratio test
            good_matches = []
            for pair in matches:
                m,n = pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            matches_all.append({"pair":(i,j),"matches":good_matches,"num_matches":len(good_matches)})
    return matches_all
