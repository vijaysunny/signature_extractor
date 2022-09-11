import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import glob

Original='/home/vijaysn/Documents/template/ip/*.*'
template='/home/vijaysn/Documents/template/template/*.*'
for file1 in glob.glob(Original):
    base1=os.path.basename(file1)
    fn1=os.path.splitext(base1)[0]
    for file2 in glob.glob(template):
        base2=os.path.basename(file2)
        fn2=os.path.splitext(base2)[0]
        if(fn1==fn2):
            img1_color=cv2.imread(file1)
            img1=cv2.imread(file1,0)
            img2=cv2.imread(file2,0)
            w, h = img2.shape[::-1]
            res = cv2.matchTemplate(img1,img2,cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where( res >= threshold)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img1_color, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
            cv2.imwrite('/home/vijaysn/Documents/template/matched/'+fn1+'.png',img1_color)

                
                
                
        

