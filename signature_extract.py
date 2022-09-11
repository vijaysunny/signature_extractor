from typing import List, Dict
import cv2
import numpy as np
import os
import glob
minLineLength = 100
maxLineGap = 50
def lines_extraction(gray: List[int]) -> List[int]:
    edges = cv2.Canny(gray, 75, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    return lines
path_ = '/home/vijaysn/Documents/title_extractor/sign_only/ip/*.*'
for file in glob.glob(path_):
    base=os.path.basename(file)
    fn=os.path.splitext(base)[0]
    image = cv2.imread(file)
    image=cv2.resize(image,(1024,1024))
    image_o=image.copy()
    image_o=cv2.cvtColor(image_o,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # converting to grayscale image
    (thresh, im_bw) = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # converting to binary image
    im_bw=cv2.GaussianBlur(im_bw,(5,5),0)
    (thresh, im_bw) = cv2.threshold(im_bw, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) 
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    im_bw=cv2.morphologyEx(im_bw,cv2.MORPH_OPEN,kernel)
    im_bw = ~im_bw
    mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create blank image of same dimension of the original image
    lines = lines_extraction(gray) # line extraction
    try:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 3)

    except TypeError:
        pass
    (contours,_) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    areas = [cv2.contourArea(c) for c in contours]
    avgArea = sum(areas)/len(areas)
    for c in contours:
        if cv2.contourArea(c)>35*avgArea:
            cv2.drawContours(mask, [c], -1, 0, -1)
    im_bw = cv2.bitwise_and(im_bw, im_bw, mask=mask) # nullifying the mask over binary
    #height heuristic
    cv2.imwrite('/home/vijaysn/Documents/title_extractor/sign_only/pre_pro/'+fn+'.png',~im_bw)
    (contours, _) = cv2.findContours(im_bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)
    # finding the larger text
    h1=1.5
    while True:
        mask = np.ones(image.shape[:2], dtype="uint8") * 255 # create the blank image

        for c in contours:
                [x,y,w,h] = cv2.boundingRect(c)
                if h >h1*avgheight and y>200:
                    cv2.drawContours(mask, [c], -1, 0, -1)
        (contours, _) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        total_contours=len(contours)
        if total_contours >2:
            h1=h1+0.5

        else:
            cv2.imwrite('/home/vijaysn/Documents/title_extractor/sign_only/mask/'+fn+'.png',mask)
            break
            #width heustic
    (contours,_) = cv2.findContours(~mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.ones(image.shape, dtype="uint8") * 255 # blank 3 layer image
    total_contours=len(contours)
    mx = (0,0,0,0)
    min_w=50
    mx_w=350
    for cont in contours:
     x,y,w,h = cv2.boundingRect(cont)
     if w >min_w and w < mx_w and y>200:
        print(fn,'->',w,h)
        boxed =  cv2.rectangle(image_o,(x,y),(x+w,y+h),(200,0,0), 2)
        title = image[y: y+h, x: x+w] 
        mask2[y: y+h, x: x+w] = title # copied title contour onto the blank image
        image[y: y+h, x: x+w] = 255 # nullified the title contour on original imag
                #cv2.imwrite('/home/vijaysn/Documents/title_extractor (copy)/sign_only/output/'+fn+'.png',image)
        cv2.imwrite('/home/vijaysn/Documents/title_extractor/sign_only/preoutput/'+fn+'.png',cv2.cvtColor(mask2,cv2.COLOR_BGR2GRAY))
        cv2.imwrite('/home/vijaysn/Documents/title_extractor/sign_only/region/'+fn+'.png',boxed)
            