import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import cv2 as cv

def contours_count(img_mask):
    img_mask=cv.GaussianBlur(img_mask,(3,3),0,0,cv.BORDER_DEFAULT)
    gray = cv.cvtColor(img_mask, cv.COLOR_RGB2GRAY)
    retval, thresh_gray = cv.threshold(gray, thresh=100, maxval=255, type=cv.THRESH_BINARY_INV)
    canny = cv.Canny(thresh_gray, 150, 120)
    kernel = np.ones((55, 55), np.uint8)
    closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    total_contours=len(contours)
    return total_contours
def green_hsv():
    low_green = np.array([25, 52, 72])
    high_green = np.array([90, 255, 255])
    green_mask = cv.inRange(hsv, low_green, high_green)
    green = cv.bitwise_and(img_rgb,img_rgb, mask=green_mask)
    total_contours=contours_count(green)
    if total_contours>2:
        green=255
        return green
    else:
        
        return green
def blue_hsv():
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    blue_mask = cv.inRange(hsv, low_blue, high_blue)
    blue = cv.bitwise_and(img_c,img_c, mask=blue_mask)
    total_contours=contours_count(blue)
    if total_contours>2:
        blue=255
        return blue
    else:
       
        return blue
def red_hsv():
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv.inRange(hsv, low_red, high_red)
    red = cv.bitwise_and(img_c,img_c, mask=red_mask)
    total_contours=contours_count(red)
    if total_contours>2:
        red=255
        return red
    else:
       
        return red
path="/home/vijaysn/Documents/color_segmentation/ip/*.*"
for file in glob.glob(path):
    img_c = cv.imread(file)
    img_c=cv.resize(img_c,(1280,1280))
    img_rgb=cv.cvtColor(img_c,cv.COLOR_BGR2RGB)
    base=os.path.basename(file)
    fn=os.path.splitext(base)[0]
    hsv = cv.cvtColor(img_rgb, cv.COLOR_RGB2HSV) 
    red=red_hsv()
    green=green_hsv()
    blue=blue_hsv()
    signature=red+green+blue
    plt.imsave("/home/vijaysn/Documents/color_segmentation/preversion1/"+fn+".png",signature)
    preversion_path="/home/vijaysn/Documents/color_segmentation/preversion1/"+fn+".png"
    result_img=cv.imread(preversion_path,0)
    result_img = cv.threshold(result_img,0,255,cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
    cv.imwrite("/home/vijaysn/Documents/color_segmentation/preoutput1/"+fn+".png", result_img)
    result_img=cv.GaussianBlur(result_img,(3,3),0,0,cv.BORDER_DEFAULT)
    #gray = cv.cvtColor(result_img, cv.COLOR_RGB2GRAY)
    retval, thresh_gray = cv.threshold(result_img, thresh=100, maxval=255, type=cv.THRESH_BINARY_INV)
    canny = cv.Canny(thresh_gray, 150, 120)
    kernel = np.ones((55, 55), np.uint8)
    closing = cv.morphologyEx(canny, cv.MORPH_CLOSE, kernel)
    contours, hierarchy = cv.findContours(closing, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for cont in contours:
            x,y,w,h = cv.boundingRect(cont)
            if(w*h >1200 and w *h < 70000):
                boxed =  cv.rectangle(img_rgb,(x,y),(x+w,y+h),(200,0,0), 2)
                plt.imsave('/home/vijaysn/Documents/color_segmentation/region1/'+fn+'.png',boxed)
