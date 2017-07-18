#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import numpy as np

#img1 = cv2.imread('../data/1010a.jpg', 0)
#declaring them globally
#sift = cv2.xfeatures2d.SURF_create(0)
#kp1, des1 = sift.detectAndCompute(img1,None)


    #order_points 
    #Formats four random cordinates passed to it for drawing a 
    #rectangle in left top, left bottom, right top, right bottom
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

    #four_point_transform 
    #Takes image and four coordinates as input. This code 
    #blocks extract the quardilatera from an image and transform into a rectangle
    #and returns that
def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped
    
    #to adjust the image size, boundary padding is done
def pad_to_square(mat):
    
    mdim = max(mat.shape[:2])
    h, w  = mat.shape[:2]

    pad_h = (mdim - h) // 2
    pad_w = (mdim - w) // 2

    padded = np.zeros((mdim, mdim), dtype=mat.dtype)
    padded[pad_h: h+pad_h, pad_w: w+pad_w] = mat.copy()

    return padded
    
    #set_h_w refers to set height width
    #this function takes input one_height, one_width, two_height,
    #two_width in the same order. In this case, size of image two is
    #larger than image one, which has to be reduced to the size of image
    #one without disturbing the aspect ration of two mage.
def set_h_w(one_h, one_w, two_h, two_w):
    if(two_h > two_w):
        two_w = int(two_w * one_h / two_h)
        two_h = one_h
    else:
        two_h = int(two_h * one_w / two_w)
        two_w = one_w

    return two_h, two_w

def image_resize(img_reference_gauss, img_warped_gauss):
    #determing shape of both the images
    ref_h, ref_w = img_reference_gauss.shape[:2]
    wrp_h, wrp_w = img_warped_gauss.shape[:2]
    
    #noteing down the max dimension size of both the images
    ref_mdim = max(ref_h, ref_w)
    wrp_mdim = max(wrp_h, wrp_w)

    #whichever image is larger in size, is reduced to the size of
    #smaller image
    if ref_mdim > wrp_mdim :
        set_h,set_w = set_h_w(wrp_h, wrp_w, ref_h, ref_w)
        img_reference_gauss = cv2.resize(img_reference_gauss, (set_w, set_h), interpolation=cv2.INTER_AREA)
    else:
        set_h,set_w = set_h_w(ref_h, ref_w, wrp_h, wrp_w)
        img_warped_gauss = cv2.resize(img_warped_gauss, (set_w, set_h), interpolation=cv2.INTER_AREA)
    
    #both the images are now padded to square
    ref_padded = pad_to_square(img_reference_gauss)
    wrp_padded = pad_to_square(img_warped_gauss)

    hR , wR = ref_padded.shape[:2]
    hW , wW = wrp_padded.shape[:2]
    
    if hR == hW and wR == wW:
    	return ref_padded,wrp_padded, 1, 1
    else:
    	return ref_padded, wrp_padded,-1,-1

    #gauss_grad
    #computes the gaussian gradient of an image and returns the gaussian 
    #gradient of that particular image. this code bloack take any image (gray scale)
    #as an input and also its name as a string
def gauss_grad (image, name): 
    result = ndimage.gaussian_gradient_magnitude(image, sigma=2)
    return result

    #normalize_img
    #This code block normalise the intensities of an image in the range
    #0-255. Note that the image should be gray scale. For example if the lowest value in
    #an image is 5 and highest is hundread, this will map 5 to 0 and 100 to 255 and all 
    #the intensities value in between accordingly. It takes an gray scale image as an 
    #input and a range(start value of range is 0 by default) and return an normalised
    #gray scale image
def normalize_img(img, drange=255):
    max_val = img.max()
    min_val = img.min()
    res = drange - ((drange*(max_val-img))/(max_val-min_val))
    return res

    #SIFT_rect
    #This code block presents an algorithm for detecting a specific object based 
    #on finding point correspondences between the reference and the target 
    #image. It can detect objects despite a scale change or in-plane 
    #rotation. It is also robust to small amount of out-of-plane rotation and 
    #occlusion. SIFT_Rect takes two image, a string and a number [0-1] as a input,
    #where first image is query image (digital reference), second image is train image
    #(wild scene), string is the name of train image, and number is threshold in range 
    #[0-1]. This code block returns a warped image taken out of the scene
    #and it also save matching result as a name_result where <name> is name of
    #your image passed as a input

    #This method of object detection works best for objects that exhibit
    #non-repeating texture patterns, which give rise to unique feature matches. 
    #This technique is not likely to work well for uniformly-colored objects, 
    #or for objects containing repeating patterns. Note that this algorithm is
    #designed for detecting a specific object. For example there are two similar 
    #object in the scene, then this will not detect all the objects
def SIFT_Rect(img1, img2, qimg, thresh):
    #initialise image = img2 for warping
    image = img2

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SURF_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
            if m.distance < thresh*n.distance:
                good.append(m)
    #set t from 0.40 to .70 for getting correct results

    if len(good)>5:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,[0,255,0],3, cv2.LINE_AA)

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                       singlePointColor = None,
                       matchesMask = matchesMask, # draw only inliers
                       flags = 2)

    pts = np.int32(dst)[:,0]

    warped = four_point_transform(image, pts)
    
    res = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    cv2.imwrite('{}_match_result.jpg'.format(qimg), res)
    cv2.namedWindow("Matching Result")
    cv2.imshow("Matching Result",res)

    return warped


bf = cv2.BFMatcher()


    #SIFT_Rect_W
    #Algorithm used in this code block is as same as the previous code block
    #This code block returns good_points which are cordinates of feature points being
    #matched in the query image with respect to the train image
def SIFT_Rect_W(img1, img2, qimg, thresh):
    
    sift = cv2.xfeatures2d.SURF_create(0)
    
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    matches = bf.knnMatch(des1,des2,k=3)

    img_R_gauss = normalize_img ( gauss_grad(img1, qimg), 2)
    img_W_gauss = normalize_img ( gauss_grad(img2, qimg), 2)
    
    good = []
    good_pts = []
    
    ct = 0

    for m,n,o in matches:
            if m.distance < thresh*n.distance:
                (x1, y1) = kp1[m.queryIdx].pt
                (x2, y2) = kp2[o.trainIdx].pt
                ct = ct +1
                if True:#img_R_gauss[y1,x1] > 0.5 and img_W_gauss[y2,x2] > 0.5:
                    good.append(m)
                    good_pts.append((int(x1),int(y1)))

    print("matches {}".format(ct))

    return good_pts

def find_defect(img1 , img2, img3, qimg, thresh):
    #from good wild image, digital reference image is being extracted
    ref_wild = qimg+"_ref_wild"
    warp_ref = SIFT_Rect(img1, img3, ref_wild, thresh)
    #in the digital_reference image, the feature points which are matched
    #are being noted down in good_one
    good_one = SIFT_Rect_W(img1, warp_ref, qimg, thresh)

    print("Level 1 over")

    #from bad wild image, digital reference image is being extracted
    wild = qimg+"_wild"
    warp_temp = SIFT_Rect(img1, img2, wild, thresh)
    #in the digital_reference image, the feature points which are matched
    #are being noted down in good_two    
    good_two = SIFT_Rect_W(img1, warp_temp, qimg, thresh)
    
    print("Level 2 over")

    #process to find out difference of features point
    #basically the feature points, which are matched in good wild image
    #but not matched in disturbed wild image
    img1_rgb = np.dstack(([img1]*3))
    temp_img = np.zeros_like(img1).astype(np.uint8)

    gs = int(0.01 * img1.shape[0] * 0.01* img1.shape[1])
    for (x,y) in good_one:
        temp_img[y-gs:y+gs, x-gs:x+gs] = 255

    for (x,y) in good_two:
        temp_img[y-gs:y+gs, x-gs:x+gs] = 0

    s = [[1,1,1], [1,1,1], [1,1,1]]
    #lbl, nlbl = ndimage.label(temp_img, s)
    #slices = ndimage.find_objects(lbl)
    #print('nblobs', nlbl)
    #for by_,bx_ in slices:
    #    by = int(by_)
    #    bx = int(bx_)
    #    if np.count_nonzero(temp_img[by.start:by.stop, bx.start:bx.stop]) <= 4*gs*gs:
    #        pass#temp_img[by.start:by.stop, bx.start:bx.stop] = 0

    lbl, nlbl = ndimage.label(temp_img, s)
    slices = ndimage.find_objects(lbl)
    print('nblobs', nlbl)
    if nlbl > 0:
        for by,bx in slices:
            img1_rgb = cv2.rectangle(img1_rgb, (bx.start, by.start), (bx.stop, by.stop), (0,255,0), 4)
    count = 0
    
    cv2.imwrite('{}_defects.jpg'.format(qimg), img1_rgb)

def alternate_diff(img_R, img_W, qimg):

    diff_img = img_W.astype(np.int) - img_R.astype(np.int)

    temp = np.zeros_like(diff_img, dtype = np.int)
    temp[diff_img>30] = diff_img[diff_img>30]
    temp = temp * 10

    return temp.astype(np.int)


def diff_approach(img1, img2, img3, qimg, thresh):
    #extracting warped images from both
    cv2.namedWindow("Dig ref image")
    cv2.imshow("Dig ref image",img1)

    cv2.namedWindow("image with problem")
    cv2.imshow("image with problem",img2)

    #qimgR is name with which matching of good wild 
    #image should be saved
    qimgR = qimg+"_ref_wild"
    ref_wild = SIFT_Rect(img1, img3, qimgR, thresh)

    #qimgW is name with which matching of bad wild 
    #image should be saved    
    qimgW = qimg+"_wild"
    wild = SIFT_Rect(img1, img2, qimgW, thresh)


    #bringing to same size to both
    img_ref_wild, img_wild, x, y = image_resize(ref_wild, wild)

    if x == -1 and y == -1:
    	print("Both images cannot be resized to same size, try different images set")
    	return 0

    #bringing their intensity map equal
    #img_wild = hist_match(img_wild , img_ref_wild)

    #img_ref_wild = cv2.erode(img_ref_wild, np.ones((7,7), dtype=np.uint8), iterations=1)
    #img_wild = cv2.erode(img_wild, np.ones((7,7), dtype=np.uint8), iterations=1)

    #calculating difference
    result = alternate_diff(img_ref_wild, img_wild, qimg)
    #result = ((img_ref_wild - img_wild)).astype(np.uint8)
    #result[result < 20] = 0
    
    cv2.namedWindow("Ref")
    cv2.imshow("Ref", img_ref_wild)

    cv2.namedWindow("wild")
    cv2.imshow("wild", img_wild)

    #result = cv2.bitwise_not(result)
    result[result>100] = -300

    result = result + img_ref_wild.astype(np.int)

    result[result<0] = 0
    
    cv2.namedWindow("result")
    cv2.imshow("result", result.astype(np.uint8))
    
    cv2.waitKey(0)