import cv2
from Tools.Convert import convert

data = convert()
matches_all = []

bf = cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck = False)
for i in range(len(data)):
    kp1,des1=data[i]

    for j in range(i+1,len(data)):
        kp2,des2 = data[j]

    matches = bf.knnMatch(des1,des2,k=2) #type: ignore
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    matches_all.append({"pair":(i,j),"Matches":good_matches})
    print(f"[ORB] Image {i} ↔ {j} → {len(good_matches)} good matches")
