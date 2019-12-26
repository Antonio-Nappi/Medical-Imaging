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
import math

rng.seed(12345)

def nothing(x):
    pass


def normalize(img):
    tmp = img-np.amin(img)
    image = tmp/np.amax(img)
    return image

def regional_mean(img,list):
    tmp = cv.blur(img,(list[0],list[1]))
    return cv.resize(tmp, (img.shape[0],img.shape[1]),interpolation=cv.INTER_LINEAR)

def normalize_and_HE(img):
    t = cv.equalizeHist(np.asarray(normalize(img),dtype=np.uint8))
    return t
    
def thresh_callback(val):
    print("Val", val)
    threshold = val
    soglia=cv.getTrackbarPos('Soglia', source_window)
    soglia=soglia*50
   
    


    ret,thres = cv.threshold(src_gray,threshold,255,0)
    
        #Opening

    kernel = np.ones((5,5),np.uint8)
    thres = cv.morphologyEx(thres, cv.MORPH_ELLIPSE, kernel)

    
    
    #canny_val=cv.getTrackbarPos('Canny Thresh', source_window)
    #canny_output = cv.Canny(src_gray, canny_val, canny_val * 2)
    
    
    _, contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    
  
    
    #I valori soglia sono stati presi dallo script 'aPriori'
    
    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > 2500 and cv.contourArea(contours[i])< 100000 ):            
            mu[i] = cv.moments(contours[i])
            
            
    # Get the mass centers
    
    mc = [None]*len(contours)
    
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        if(cv.contourArea(contours[i]) > 2500 and cv.contourArea(contours[i])< 100000):            
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
            print("Lungh %.2f, mass number %.2f " % (cv.arcLength(contours[i], True),i))
            '''
            #Se trovo regioni molto lunghe aumento il thresh di 3
            if( cv.arcLength(contours[i], True) > 800):
                cv.setTrackbarPos('Thresh',source_window,val+3)
                thresh_callback(val+3)
            '''

    # Draw contours
    
    #Questo crea un'altra immagine con solo le parti evidenziate, buono per fare la maschera
    drawing = np.zeros((thres.shape[0], thres.shape[1], 3), dtype=np.uint8)
    
    #Mi copio l'immagine originale in modo tale da salvarmi l'originale,
    #altrimenti aggiungiavamo sempre più drawing
    src2=src.copy()
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > 2500 and cv.contourArea(contours[i])< 100000):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(src2, contours, i, color, 2)
            cv.circle(src2, (int(mc[i][0]), int(mc[i][1])), 20, color, -1)
            
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > 2500 and cv.contourArea(contours[i])< 100000):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color, 2)
    
    src2 = cv.resize(src2,(400,500))
    drawing = cv.resize(drawing,(400,500))
    thres = cv.resize(thres,(400,500))
    cv.imshow('Mask', drawing)
    cv.imshow(source_window, src2)
    cv.imshow('Thres', thres)


    
    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > 2500 and cv.contourArea(contours[i])< 100000):            
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))

source="20588334_493155e17143edef_MG_L_CC_ANON.tif"
src = cv.imread(source)

if src is None:
    print('Could not open or find the image:', source)
    exit(0)
#src = cv.resize(src,(400,500))

source_window = 'Source'

cv.namedWindow(source_window)
src2 = cv.resize(src,(400,500))
cv.imshow(source_window, src2)
max_thresh = 255
thresh = 100 # initial threshold
cv.createTrackbar('Thresh', source_window, thresh, max_thresh, thresh_callback)
#cv.createTrackbar('Canny Thresh:', source_window, thresh, max_thresh, nothing)
cv.createTrackbar('Soglia', source_window, thresh, 1000, nothing)



src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)


img2 = normalize(src_gray)

clahe = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
img_clahe = clahe.apply(src_gray)
mu = np.average(src_gray)
tmp = normalize(img_clahe)
img3 = np.zeros((tmp.shape[0],tmp.shape[1]))
for i in range(img_clahe.shape[0]):
    for j in range(img_clahe.shape[1]):
        img3[i,j]= tmp[i,j] * (1-math.exp(-(img2[i,j]/mu)))
img3 = normalize(img3)
m2 = regional_mean(img3,[16,16])
img4 = normalize_and_HE(img3)
img5 = normalize(img3 + regional_mean(m2,[16,16]))
img6 = normalize_and_HE(img5) * img4
e = normalize(img6)    
    
    
    
src=img5.copy()
# Convert image to gray and blur it

'''
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
src_gray = cv.blur(src_gray, (3,3))
'''

thresh_callback(thresh)
cv.waitKey()