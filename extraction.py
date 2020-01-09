# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 18:45:57 2020

@author: AntonioBho
"""


from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import random as rng
import os
from utils import aPriori


def control(contours,ground_path):
    #Dati ricavati dal file aPriori
    min_area,average_area,max_area,min_perimeter,average_perimeter,max_perimeter=aPriori.extract_information(ground_path)
    return cv.contourArea(contours) > min_area and cv.contourArea(contours)< max_area and cv.arcLength(contours, True) > min_perimeter and cv.arcLength(contours, True)< max_perimeter

def control2(contours):
    #Se non trova nessun contorno restituisce false
    if (len (contours)<1):
        return False
    else:
        for i in range(len(contours)):
            #se la funzione control restituisce true vuol dire che almeno un contorno va bene quindi restituisco true
            if(control(contours[i])):
                return True
        #Se alla fine del ciclo nessun contorno va bene allora restituisce False
        return False

def thresh_callback(val):
            threshold = val
            print(threshold)
            
            
           
            #   Threshold
            ret,thres = cv.threshold(src,threshold,255,0,cv.THRESH_BINARY+cv.THRESH_OTSU)
            
           
            #   Opening
        
            kernel = np.ones((5,5),np.uint8)
            thres = cv.morphologyEx(thres, cv.MORPH_ELLIPSE, kernel)
            
            #   Contours
            _, contours, _ = cv.findContours(thres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
            #I valori soglia sono stati presi dallo script 'aPriori'
        
            #Se non trovo masse abbasso il valore di threshold a 105
            if not control2(contours):
            
                thresh_callback(105)
            else:
                    # Get the moments
                    mu = [None]*len(contours)
                    for i in range(len(contours)):
                        if(control(contours[i])):            
                            mu[i] = cv.moments(contours[i])
                
                    # Get the mass centers
                    
                    mc = [None]*len(contours)
                    
                    for i in range(len(contours)):
                        # add 1e-5 to avoid division by zero
                        if(control(contours[i])):            
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
                        if(control(contours[i])):            
                            color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
                            cv.drawContours(src2, contours, i, color,2)
                            # Il punto centrale è localizzato da -> (int(mc[i][0]), int(mc[i][1]))
                           # cv.circle(src2, (int(mc[i][0]), int(mc[i][1])), 5, color, -1)
                           
        
                            
                    for i in range(len(contours)):
                        if(control(contours[i])):            
                            cv.drawContours(drawing, contours, i, (255,255,255), -1)#-1 = FILLED
                        
        
                    # Visualizzazione
                    cv.imshow('Mask', drawing)
                    mask_name=name[:-3]
                    cv.imwrite("OutputExtraction\masks\\" + mask_name+"mask.png", drawing)

                
                    font= cv.FONT_HERSHEY_COMPLEX
                    src2 = cv.putText(src2, name, (0,10), font,0.5, (250,0,255))
                
                    cv.imshow(source_window, src2)
                    cv.imwrite("OutputExtraction\\" + name, src2)

                    #cv.imshow('Thres', thres)
                    cv.imwrite("OutputUnet\\" + name, src)

                
                    # Calculate the area with the moments 00 and compare with the result of the OpenCV function
                    for i in range(len(contours)):
                        if(control(contours[i])):            
                            print(' * Contour[%d] - Area (M_00) = %.2f - Area OpenCV: %.2f - Length: %.2f' % (i, mu[i]['m00'], cv.contourArea(contours[i]), cv.arcLength(contours[i], True)))
                            
        
data = np.loadtxt("predictions.txt",dtype = np.float64)
print(data.shape) #Dallo shape faccio diviso 512 e capisco quante immagini ho, e ottengo 45 sub_array
lista=np.split(data,45)

#Per ottenere il nome dell'immagine
test_path="test_enhanced"
test_images= os.listdir(test_path)


i=0
for test in test_images:
        name=test
        print(test_path+"\\"+test)
        mask=cv.imread(test_path+"\\"+test,cv.IMREAD_ANYDEPTH)
        if mask is None:
            print('Could not open or find the image:', mask)
        mask=cv.resize(mask,(512,512)) 
        mask=mask*255
        mask=mask.astype('uint8')
        
        #Erosione sulla maschera per rimuovere i bordi
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7,7))
        mask = cv.erode(mask, kernel, iterations=3)
        
        ret,mask = cv.threshold(mask,1,255,0,cv.THRESH_BINARY+cv.THRESH_OTSU)
        
        
        src = lista[i]
        
        src=src*255
        #♥Converto per il findContours che richiede solo file di uint8 o al massimo 32
        src = src.astype('uint8')
        #Salvo l'immagine
        cv.imwrite("OutputUnet\\" + name, src)

        #Applico la maschera
        src[mask==0]=0        
                
        source_window = 'Source'
        
        cv.namedWindow(source_window)
        cv.imshow(source_window, src)
        max_thresh = 255
        thresh = 122 # initial threshold
        #cv.createTrackbar('Thresh', source_window, thresh, max_thresh, thresh_callback)
        
        thresh_callback(thresh)
        cv.waitKey()
        i+=1
        print("Immagine n " , i)
