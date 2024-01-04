from data.dataset import readIndex, dataReadPip, loadedDataset
from model.deepcrack import DeepCrack
from config import Config as cfg
from trainer import DeepCrackTrainer
import cv2
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch
import os
import linecache
import time
import math
import heatmap
import matplotlib.pylab as plt
import glob
import itertools
import argparse
from plantcv import plantcv as pcv
import numpy as np
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage import measure
from skimage.segmentation import random_walker
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import argparse
import random as rng
import cv2
import numpy as np
import random
import sys
import math
import collections
from math import acos
from math import sqrt
from math import pi

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner


strFilePath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/2.bmp'
strSaveFilePath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result.jpg'
img = cv2.imread(strFilePath)
#img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

source_window = 'Image'
maxTrackbar = 100
rng.seed(12345)

def leafDetection():    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #blur= cv2.GaussianBlur(gray, (5, 5), 2)

    mono = cv2.threshold(gray, 48, 255, cv2.THRESH_BINARY_INV)[1]

    mono = cv2.erode(mono, (3,3), iterations=2)

    mono = cv2.dilate(mono, (3,3), iterations=3)
    
    color_01 = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    color_02 = cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)

    src = img
    dst = src.copy()  
    gray_line = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    label = cv2.connectedComponentsWithStats(mono)
    mono = cv2.dilate(mono, (3,3), iterations=10)
    n = label[0] - 1
    data = np.delete(label[2], 0, 0)
    center = np.delete(label[3], 0, 0)    
    for i in range(n):
        Point2D = collections.namedtuple('Point2D', ['x', 'y'])    # namedtuple로 점 표현 
        #leftTop
        xPoint0 =1
        yPoint0 =1
        xPoint1 =1
        yPoint1 =1
        bCheck =False
        x0 = data[i][0]
        y0 = data[i][1]                 
        #right Bottom
        x1 = data[i][0] + data[i][2]
        y1 = data[i][1] + data[i][3]
        
        #left bottom
        x2 = data[i][0]
        y2 = data[i][1]+ data[i][3]
        
        #right top
        x3 = data[i][0]+ data[i][2]
        y3 = data[i][1]      

        p1 = Point2D(x=xPoint0, y=yPoint0)    # 점1
        p2 = Point2D(x=xPoint1, y=yPoint1)    # 점2
        
        a = p1.x - p2.x    # 선 a의 길이
        b = p1.y - p2.y    # 선 b의 길이
        
        c = math.sqrt((a * a) + (b * b))
        c = round(c)

        angle = angle_clockwise(p1, p2)
        angle = round(angle)
        

        cv2.rectangle(color_01, (x0, y0), (x1, y1), (0, 0, 255))
        cv2.rectangle(color_02, (x0, y0), (x1, y1), (0, 0, 255))
        #cv2.putText(color_01, "ID: " +str(i + 1), (x1 - 20, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        #cv2.putText(color_01, "S: " +str(data[i][4]), (x1-60, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        #cv2.putText(color_01, "L: " + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255))        
        cv2.putText(color_01, "len: " +str(c), (x0, y0), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.putText(color_01, "angle: " +str(angle), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

        cv2.putText(color_02, "X: " + str(int(center[i][0])), (x1 - 30, y1 + 15), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv2.putText(color_02, "Y: " + str(int(center[i][1])), (x1 - 30, y1 + 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))        

   
    cv2.imshow("color_01", color_01)
    cv2.imshow("color_02", color_02)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
def goodFeaturesToTrack_Demo(val):
    maxCorners = max(val, 1)
    # Parameters for Shi-Tomasi algorithm
    qualityLevel = 0.21
    minDistance = 10
    blockSize = 3
    gradientSize = 3
    useHarrisDetector = False
    k = 0.04
    # Copy the source image
    copy = np.copy(src)
    # Apply corner detection
    corners = cv.goodFeaturesToTrack(src_gray, maxCorners, qualityLevel, minDistance, None, \
        blockSize=blockSize, gradientSize=gradientSize, useHarrisDetector=useHarrisDetector, k=k)
    # Draw corners detected
    print('** Number of corners detected:', corners.shape[0])
    radius = 4
    for i in range(corners.shape[0]):
        cv.circle(copy, (int(corners[i,0,0]), int(corners[i,0,1])), radius, (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256)), cv.FILLED)
    # Show what you got
    cv.namedWindow(source_window)
    cv.imshow(source_window, copy)


def distanceTransforom_test():
    # Sample study area array
    strFilePath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/test.jpg'
    strSaveFilePath = 'D:/DeepCrack-master/DeepCrack-master/codes/deepcrack_results/result.jpg'
    img = cv2.imread(strFilePath)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # binaray image로 변환
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    #Morphology의 opening, closing을 통해서 노이즈나 Hole제거
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)

    # dilate를 통해서 확실한 Backgroud
    sure_bg = cv2.dilate(opening,kernel,iterations=1)

    #distance transform을 적용하면 중심으로 부터 Skeleton Image를 얻을 수 있음.
    # 즉, 중심으로 부터 점점 옅어져 가는 영상.
    # 그 결과에 thresh를 이용하여 확실한 FG를 파악
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)
    sure_fg = np.uint8(sure_fg)

    # Background에서 Foregrand를 제외한 영역을 Unknow영역으로 파악
    unknown = cv2.subtract(sure_bg, sure_fg)

    # FG에 Labelling작업
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # watershed를 적용하고 경계 영역에 색지정
    markers = cv2.watershed(img,markers)
    img[markers == -1] = [255,0,0]


    images = [gray,thresh,sure_bg,  dist_transform, sure_fg, unknown, markers, img]
    titles = ['Gray','Binary','Sure BG','Distance','Sure FG','Unknow','Markers','Result']

    for i in range(len(images)):
        plt.subplot(2,4,i+1),plt.imshow(images[i]),plt.title(titles[i]),plt.xticks([]),plt.yticks([])

    plt.show()

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))    
    
    blank_ch = 255*np.ones_like(label_hue)
    
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    cv2.imshow('labeled.png', labeled_img)
    cv2.waitKey()


if __name__ == '__main__':    
     
    # img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]  # ensure binary
    # num_labels, labels_im = cv2.connectedComponents(img)

    #imshow_components(labels_im)
    leafDetection()
  
  
    



