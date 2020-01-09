# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 15:03:13 2020
Jaccard
"""

import cv2 as cv
import numpy as np
import os
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import jaccard_score

def jaccard_similarity(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())


true_path="mask_true"
true_dir=os.listdir(true_path)

pred_path="OutputExtraction\masks"
pred_dir=os.listdir(pred_path)

i=0
somma=0
for i in range(len(pred_dir)):
    true=true_dir[i]
    pred=true_dir[i]
    
    if(true==pred):
        true_img=cv.imread(true_path+"\\"+true,cv.IMREAD_ANYDEPTH)
        true_img=cv.resize(true_img,(512,512)) 

        pred_img=cv.imread(pred_path+"\\"+pred,cv.IMREAD_ANYDEPTH)
        
        #img_true=np.array(true_img).ravel()
        #img_pred=np.array(pred_img).ravel()
        
        
        img_true = np.asarray(true_img).astype(np.bool)
        img_pred = np.asarray(pred_img).astype(np.bool)
       # iou = jaccard_similarity_score(img_true, img_pred)
        
        print(type(img_true)) 
        print(img_true)
        #cv.imshow("IMG_TRUE",img_true)
        #cv.imshow("IMG_PRED",img_pred)
       # print("Jaccard n "+str(i),iou)
        jaccard=jaccard_similarity(img_true,img_pred)
        #jaccard=jaccard_score(img_true,img_pred,None,1, average='weighted')
        print("Jaccard Score n "+str(i),jaccard)
        somma=somma+jaccard

media=somma/(i+1)
print("MEDIA : ",media)
cv.waitKey()