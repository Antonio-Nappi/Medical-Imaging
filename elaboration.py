# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 14:17:50 2019

@author: AntonioBho
"""
import cv2 as cv
import numpy as np
import random as rng
import os
import time
import copy
rng.seed(time.time())

def nothing(x):
    pass
'''
def thresh_callback(val):
    threshold = val
    

   
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
'''


#def find_countours(mass_path,Amax=750000,Amin=2500):
mass_path = "dataset\enhanced\\"
mass_images = os.listdir(mass_path)

for mass in mass_images:
    print(mass_path+mass)
    src = cv.imread(mass_path + mass,cv.IMREAD_ANYDEPTH)
    print(np.unique(src))
    hist, bin=np.histogram(src)
    print(bin)
    cv.namedWindow("th")
    width,height=src.shapex
    tmp = copy.deepcopy(src)
    hist = cv.calcHist([src],[0],None,[0. ,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],[0.,1.])
    #primo tentativo val:  9.231093e-16
    # primo tentativo val:  9.595606e-16
    # primo tentativo val:  9.725499e-16
    # primo tentativo val:  1.0173351e-15
    # primo tentativo val:  1.1688327e-15
    # primo tentativo val:  0.038179856
    # primo tentativo val:  0.048218753
    # primo tentativo val:  0.059720024
    # primo tentativo val:  0.06377587
    # primo tentativo val:  0.06511031
    # primo tentativo val:   0.06548106
    # primo tentativo val:  0.065624475
    # primo tentativo val:  0.06567243
    cv.createTrackbar("TH", "th", 100, 600, nothing)

    while (1):  # 0-NERO   1-BIANCO #0.3981747604413089458
        prefix = 0.0001747604413089458
        th = (cv.getTrackbarPos("TH", "th")) + 100
        val = (th / 1000) + prefix
        print("val: %8.55f" % (val))
        ret, th_img = cv.threshold(src, 0.6567243, 1, cv.THRESH_BINARY)
        pix2 = cv.countNonZero(th_img)
        cv.imshow("th", np.hstack((cv.resize(src, (600, 500)), cv.resize(th_img, (600, 500)))))
        cv.imshow("histogram",hist)
        key = cv.waitKey(1)
        if key == 27:
            break
    '''max_thresh = 255
    thresh = 100 # initial threshold
    cv.createTrackbar('Thresh', source_window, thresh, max_thresh, thresh_callback)
    
    src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    
    img5, img4= nappi(src_gray)
    
    img6 = normalize_and_HE(img5) * img4
    e = normalize(img6)
    src=img5.copy()
    thresh_callback(thresh)
    cv.waitKey()
    '''