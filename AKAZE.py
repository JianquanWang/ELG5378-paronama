import cv2
import PIL.Image as Image
import numpy as np
import time

def warpTwoImages(img1, img2, H):
    '''warp img2 to img1 with homograph H'''
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, Ht.dot(H), (xmax-xmin, ymax-ymin))
    result[t[1]:h1+t[1],t[0]:w1+t[0]] = img1
    return result


def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
                   ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    rawMatches = sorted(rawMatches, key=lambda x: x[0].distance)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
                                         reprojThresh)

        return (matches, H, status)

    return None

def matchKeypoints_(kpsA, kpsB, featuresA, featuresB,
                   ratio, reprojThresh):
    # compute the raw matches and initialize the list of actual
    # matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    print(featuresA.dtype)
    rawMatches = bf.knnMatch(featuresA, featuresB, k=2)
    rawMatches = sorted(rawMatches, key=lambda x: x[0].distance)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,
                                         reprojThresh)

        return (matches, H, status)

    return None

def kaze_match(im1_path, im2_path):
    # load the image and convert it to grayscale
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)


    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    print(descs1.dtype)
    # Match the features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(descs1,descs2, k=2)    # typo fixed

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3 = cv2.drawMatchesKnn(im1, kps1, im2, kps2, good[:], None, flags=2)
    #cv2.imshow("AKAZE matching", im3)
    result = Image.fromarray(im3)
    #result.show()
    cv2.waitKey(0)
    return im3


def sift_match(im1_path, im2_path):
    # load the image and convert it to grayscale
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.xfeatures2d.SIFT_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)


    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    # Match the features
    matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
    matches = matcher.knnMatch(descs1, descs2, 2)
    matches = sorted(matches, key=lambda x: x[0].distance)
    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.9*n.distance:
            good.append([m])

    # cv2.drawMatchesKnn expects list of lists as matches.
    im3 = cv2.drawMatchesKnn(im1, kps1, im2, kps2, good[:], None, flags=2)
    #cv2.imshow("AKAZE matching", im3)
    result = Image.fromarray(im3)
    result.show()
    cv2.waitKey(0)
    return im3

def sift_stitch(im1, im2):
    # load the image and convert it to grayscale

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.xfeatures2d.SIFT_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    # convert keypoint to numpy
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])


    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    M = matchKeypoints(kps1, kps2, descs1, descs2, 0.75, 5)


    if M is None:
        print("no enough matched")
        return None
    (matches, H, status) = M

    #print(H)

    result = warpTwoImages(im1, im2, H)

    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2BGRA)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # result[3] =150
    # im2[3] = 150
    #result[0:im2.shape[0], 0:im2.shape[1]] = im2

    #im3 = Image.fromarray(result)
    #im3.show()
    return result

def kaze_stitch(im1, im2):
    # load the image and convert it to grayscale

    gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # initialize the AKAZE descriptor, then detect keypoints and extract
    # local invariant descriptors from the image
    detector = cv2.AKAZE_create()
    (kps1, descs1) = detector.detectAndCompute(gray1, None)
    (kps2, descs2) = detector.detectAndCompute(gray2, None)

    # convert keypoint to numpy
    kps1 = np.float32([kp.pt for kp in kps1])
    kps2 = np.float32([kp.pt for kp in kps2])


    print("keypoints: {}, descriptors: {}".format(len(kps1), descs1.shape))
    print("keypoints: {}, descriptors: {}".format(len(kps2), descs2.shape))

    #M = matchKeypoints(kps1, kps2, descs1, descs2, 0.75, 5)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    print(descs1.dtype)
    rawMatches = bf.knnMatch(descs1, descs2, k=2)
    rawMatches = sorted(rawMatches, key=lambda x: x[0].distance)
    matches = []
    # loop over the raw matches
    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    # computing a homography requires at least 4 matches
    if len(matches) > 4:
        # construct the two sets of points
        ptsA = np.float32([kps1[i] for (_, i) in matches])
        ptsB = np.float32([kps2[i] for (i, _) in matches])
        # compute the homography between the two sets of points
        (H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,5)

    M = (matches, H, status)

    if M is None:
        print("no enough matched")
        return None
    (matches, H, status) = M

    #print(H)

    result = warpTwoImages(im1, im2, H)

    # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2BGRA)
    # result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    # result[3] =150
    # im2[3] = 150
    #result[0:im2.shape[0], 0:im2.shape[1]] = im2

    #im3 = Image.fromarray(result)
    #im3.show()
    return result


im1_path = "./car/wall1.jpg"
im2_path = "./car/wall2.jpg"
im3_path = "./car/wall3.jpg"
im4_path = "./car/wall4.jpg"
im5_path = "./car/wall5.jpg"
im6_path = "./car/wall6.jpg"
im1 = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)
im3 = cv2.imread(im3_path)
im4 = cv2.imread(im4_path)
im5 = cv2.imread(im5_path)
im6 = cv2.imread(im6_path)

def test_kaze_match():

    result1 = kaze_match(im1_path, im2_path)
    cv2.imwrite("./car/kaze_match_1.jpg", result1)

    result2 = kaze_match(im2_path, im3_path)
    cv2.imwrite("./car/kaze_match_2.jpg", result2)

    result3 = kaze_match(im3_path, im4_path)
    cv2.imwrite("./car/kaze_match_3.jpg", result3)

    result4 = kaze_match(im4_path, im5_path)
    cv2.imwrite("./car/kaze_match_4.jpg", result4)

#test_kaze_match()

def test_sift_match():

    result1 = sift_match(im1_path, im2_path)
    cv2.imwrite("./car/sift_match_1.jpg", result1)

    result2 = sift_match(im2_path, im3_path)
    cv2.imwrite("./car/sift_match_2.jpg", result2)

    result3 = sift_match(im3_path, im4_path)
    cv2.imwrite("./car/sift_match_3.jpg", result3)

    result4 = sift_match(im4_path, im5_path)
    cv2.imwrite("./car/sift_match_4.jpg", result4)

#test_sift_match()
#sift_match(im1_path, im2_path)

#result = sift_stitch(sift_stitch(sift_stitch(sift_stitch(im1, im2),im3),im4),im5)
def test_sift():
    result1 = sift_stitch(im1, im2)


    cv2.imwrite("./car/sift_paronomic_result.jpg", result1)
    #result = Image.fromarray(result1)
    #result.show()
def test_akaze():
    result1 = kaze_stitch(im1, im2)


    cv2.imwrite("./car/akaze_paronomic_result.jpg", result1)
    #result = Image.fromarray(result1)
    #result.show()
#kaze_match(im1_path,im2_path)
tic1 = time.time()
test_sift()
tic2 = time.time()
test_akaze()
tic3 = time.time()
print("SIFT cost",tic2-tic1, "s")
print("AKAZE cost", tic3-tic2, "s")