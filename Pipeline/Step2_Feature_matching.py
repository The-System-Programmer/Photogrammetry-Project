import cv2 as cv

def matching():
    from Pipeline.Step1_Feature_extraction import features
    features = features()
    good_matches = []
    for i in range(len(features)-1):
        image1 = features[i]
        image2 = features[i+1]
        print(f"Feature matching between {image1["path"]} and {image2["path"]}")

        bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=False)
        matches = bf.knnMatch(image1["ds"],image2["ds"],k=2)
        for m,n in matches:
            if m.distance <0.75 * n.distance:
                good_matches.append(m)
    return good_matches
