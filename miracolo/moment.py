# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:17:50 2019

@author: AntonioBho
"""

from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import random as rng
rng.seed(12345)

def nothing(x):
    pass
    
def thresh_callback(val):
    threshold = val
    soglia=cv.getTrackbarPos('Soglia', source_window)
   
    ret,thres = cv.threshold(src_gray,threshold,255,0)
    #canny_val=cv.getTrackbarPos('Canny Thresh', source_window)
    #canny_output = cv.Canny(src_gray, canny_val, canny_val * 2)
    
    
    _, contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        if((cv.arcLength(contours[i], True)) > soglia):            
            mu[i] = cv.moments(contours[i])
            
            
    # Get the mass centers
    
    mc = [None]*len(contours)
    
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        if((cv.arcLength(contours[i], True)) > soglia):            
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            print("Lungh %.2f, mass number %.2f " % (cv.arcLength(contours[i], True),i))

    # Draw contours
    
    #Questo crea un'altra immagine con solo le parti evidenziate, buono per fare la maschera
    drawing = np.zeros((thres.shape[0], thres.shape[1], 3), dtype=np.uint8)
    
    #Mi copio l'immagine originale in modo tale da salvarmi l'originale,
    #altrimenti aggiungiavamo sempre più drawing
    src2=src.copy()
    for i in range(len(contours)):
        if((cv.arcLength(contours[i], True)) > soglia):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(src2, contours, i, color, 2)
            cv.circle(src2, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
            
    for i in range(len(contours)):
        if((cv.arcLength(contours[i], True)) > soglia):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color, 2)
    
    cv.imshow('Mask', drawing)
    cv.imshow(source_window, src2)

    
    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        if((cv.arcLength(contours[i], True)) > soglia):            
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))

source="esempio2.tif"
src = cv.imread(source)
src = cv.resize(src,(400,500))

if src is None:
    print('Could not open or find the image:', source)
    exit(0)
# Convert image to gray and blur it
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
source_window = 'Source'
cv.namedWindow(source_window)
cv.imshow(source_window, src)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Thresh:', source_window, thresh, max_thresh, thresh_callback)
#cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, nothing)
cv.createTrackbar('Soglia', source_window, thresh, 1000, nothing)

thresh_callback(thresh)
cv.waitKey()