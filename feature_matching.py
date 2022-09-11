import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
import glob
Original_path='/home/vijaysn/Documents/sift/ip/*.*'
template_path='/home/vijaysn/Documents/sift/template/*.*'
for file1 in glob.glob(template_path):
    base1=os.path.basename(file1)
    fn1=os.path.splitext(base1)[0]
    for file2 in glob.glob(Original_path):
        base2=os.path.basename(file2)
        fn2=os.path.splitext(base2)[0]
        if(fn1==fn2):
            img1_bgr = cv.imread(file1,cv.IMREAD_COLOR)    
            img1_rgb=cv.cvtColor(img1_bgr,cv.COLOR_BGR2RGB)
            img1 = cv.cvtColor(img1_rgb,cv.COLOR_RGB2GRAY)
            img2_bgr = cv.imread(file2,cv.IMREAD_COLOR)
            img2_rgb=cv.cvtColor(img2_bgr,cv.COLOR_BGR2RGB) 
            img2_rgb=cv.resize(img2_rgb,(1024,1024)) 
            img2 = cv.cvtColor(img2_rgb,cv.COLOR_RGB2GRAY)
            # Initiate SIFT detector
            sift = cv.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50)   # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params,search_params)
            matches = flann.knnMatch(des1,des2,k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = [[0,0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            good = []
            for i,(m,n) in enumerate(matches):
                if m.distance < 0.7*n.distance:
                    matchesMask[i]=[1,0]
                    good.append(m)

            MIN_MATCH_COUNT = 5
            if len(good)>=MIN_MATCH_COUNT:
                src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
                M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
                matchesMask = mask.ravel().tolist()
                pts = src_pts[mask==1]
                min_x, min_y = np.int32(pts.min(axis=0))
                max_x, max_y = np.int32(pts.max(axis=0))
                h,w = img1.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv.perspectiveTransform(pts,M)
                img2 = cv.polylines(img2_rgb,[np.int32(dst)],True,255,3, cv.LINE_AA)
            else:
                print( "Not enough matches are found - {},{}/{}".format(fn1,len(good), MIN_MATCH_COUNT) )
                matchesMask = None

            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
                            
            img3 = cv.drawMatches(img1_rgb,kp1,img2_rgb,kp2,good,None,**draw_params)
            #plt.imshow(img3),plt.show()
            cv.imwrite('/home/vijaysn/Documents/sift/matched/'+fn1+'.png', img3)
            cv.rectangle(img1_rgb,(min_x, min_y), (max_x,max_y), 255,2)
            cv.imwrite('/home/vijaysn/Documents/sift/region/'+fn1+'.png',img2_rgb)
