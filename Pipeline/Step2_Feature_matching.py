import cv2 as cv

def matching(features):
    good_matches = []
    pair_matches = []
    for i in range(len(features)-1):
        image1 = features[i]
        image2 = features[i+1]
        print(f"Feature matching between {image1["path"]} and {image2["path"]}")
        bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(image1["ds"],image2["ds"],k=2)
        for m,n in matches:
            if m.distance <0.75 * n.distance:
                pair_matches.append(m)
            good_matches.append({
                "img1":image1["path"],
                "img2":image2["path"],
                "matches":pair_matches
            })
    return pair_matches
