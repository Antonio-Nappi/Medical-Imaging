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
import os
import time
rng.seed(time.time())

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

def preprocessing(src_gray):
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
    return img5,img4
    
def thresh_callback(val):
    threshold = val
    
    soglia_max=750000
    soglia_min=2500
   
    #   Threshold
    ret,thres = cv.threshold(src_gray,threshold,255,0)
    
    #   Opening

    kernel = np.ones((5,5),np.uint8)
    thres = cv.morphologyEx(thres, cv.MORPH_ELLIPSE, kernel)
    
    #   Contours
    _, contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #I valori soglia sono stati presi dallo script 'aPriori'

    # Get the moments
    mu = [None]*len(contours)
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > soglia_min and cv.contourArea(contours[i])< soglia_max ):            
            mu[i] = cv.moments(contours[i])

    # Get the mass centers
    
    mc = [None]*len(contours)
    
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        if(cv.contourArea(contours[i]) > soglia_min and cv.contourArea(contours[i])< soglia_max):            
            mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
                
                #print(' * MC[%d] =' % (i) )
                #print(mc[i])
            
    # Draw contours
    #Questo crea un'altra immagine con solo le parti evidenziate, buono per fare la maschera
    drawing = np.zeros((thres.shape[0], thres.shape[1], 3), dtype=np.uint8)
    
    #Mi copio l'immagine originale in modo tale da salvarmi l'originale,
    #altrimenti aggiungiavamo sempre più drawing
    src2=src.copy()
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > soglia_min and cv.contourArea(contours[i])< soglia_max):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(src2, contours, i, color, 2)
            cv.circle(src2, (int(mc[i][0]), int(mc[i][1])), 20, color, -1)
            
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > soglia_min and cv.contourArea(contours[i])< soglia_max):            
            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
            cv.drawContours(drawing, contours, i, color, 2)

    # Visualizzazione
    src2 = cv.resize(src2,(400,500))
    drawing = cv.resize(drawing,(400,500))
    thres = cv.resize(thres,(400,500))
    cv.imshow('Mask', drawing)

    font= cv.FONT_HERSHEY_COMPLEX
    src2 = cv.putText(src2, name, (0,10), font,0.5, (250,0,255))

    cv.imshow(source_window, src2)
    cv.imshow('Thres', thres)

    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
    for i in range(len(contours)):
        if(cv.contourArea(contours[i]) > soglia_min and cv.contourArea(contours[i])< soglia_max):            
            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))

mass_path = "dataset\images\mass"
mass_images = os.listdir(mass_path)

for mass in mass_images:
    
        name=mass
        # read the training image
        src = cv.imread("dataset\images\mass\\" + mass, cv.IMREAD_COLOR)

#source="esempio.tif"
#src = cv.imread(source)
        
        if src is None:
            print('Could not open or find the image:', src)
            exit(0)
        source_window = 'Source'
        
        cv.namedWindow(source_window)
        src2 = cv.resize(src,(400,500))
        cv.imshow(source_window, src2)
        max_thresh = 255
        thresh = 100 # initial threshold
        cv.createTrackbar('Thresh', source_window, thresh, max_thresh, thresh_callback)
        
        src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        
        img5, img4= nappi(src_gray)

        img6 = normalize_and_HE(img5) * img4
        e = normalize(img6)
        src=img5.copy()
        thresh_callback(thresh)
        cv.waitKey()