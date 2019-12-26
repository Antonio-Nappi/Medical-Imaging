from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import random as rng
import math
import os




def aPriori(image,num):

    _, contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    list_area = []
    list_lung = []
    sum_area=0
    sum_lung=0
    for i in range(len(contours)):
        print("Image = "+ str(num) + " , Contours = " + str(i+1)+ " Ground_images = "+ str(num-1))
        #L'if è stato utilizzato per evitare i pixel isolati o piccolissime regioni di pixel che non fanno parte della massa
        if(cv.contourArea(contours[i])>50 and cv.arcLength(contours[i], True)> 40):
            list_area.append(cv.contourArea(contours[i]))
            list_lung.append(cv.arcLength(contours[i], True))
            sum_area+= cv.contourArea(contours[i])
            sum_lung+= cv.arcLength(contours[i], True)
    
    
    
    return list_area, list_lung,sum_area,sum_lung
        







ground_path = "dataset\groundtruth"
ground_images = os.listdir("dataset\groundtruth")


list_area_tot=[]
list_lung_tot=[]
sum_area_tot=0
sum_lung_tot=0


num=1

'''
img = cv.imread("dataset\groundtruth\\20588046_024ee3569b2605dc_MG_R_ML_ANON.tif", cv.IMREAD_GRAYSCALE)
if img is None:
        print('Could not open or find the image:', img)
        exit(0)

list_area_tot,list_lung_tot=aPriori(img,num)


print("AREA", list_area_tot)

'''
for ground in ground_images:
    img = cv.imread(ground_path +"\\"+ ground, cv.IMREAD_GRAYSCALE)
    if img is None:
        print('Could not open or find the image:', img)
        exit(0)
    list_Area,list_Lung,sum_Area,sum_Lung = aPriori(img,num)
    sum_area_tot+=sum_Area
    sum_lung_tot+=sum_Lung
    
    list_area_tot.extend(list_Area)
    list_lung_tot.extend(list_Lung)
    num+=1
    
valore_medio_area= sum_area_tot/len(list_area_tot)
valore_medio_lung=sum_lung_tot/len(list_lung_tot)
valore_min_area=min(list_area_tot)
valore_min_lung=min(list_lung_tot)
valore_max_area=max(list_area_tot)
valore_max_lung=max(list_lung_tot)
   
for i in range(len(list_area_tot)):
  print(' * Area[%d] = %.2f, Lung[%d] = %.2f' % (i, list_area_tot[i],i, list_lung_tot[i]))

print("Valore Minimo per l'Area : ", valore_min_area)  
print("Valore Medio per l'Area : ", valore_medio_area)
print("Valore Massimo per l'Area : ", valore_max_area)

print("Valore Minimo per la Lunghezza : ", valore_min_lung)
print("Valore Medio per la Lunghezza : ", valore_medio_lung)
print("Valore Massimo per la Lunghezza : ", valore_max_lung)


''' 
OUTPUT
Valore Minimo per l'Area :  2995.5
Valore Medio per l'Area :  97040.43534482758
Valore Massimo per l'Area :  751132.0

Valore Minimo per la Lunghezza :  216.46803629398346
Valore Medio per la Lunghezza :  1303.7378783071863
Valore Massimo per la Lunghezza :  3855.347655892372



'''
  
  
  

img=cv.resize(img,(400,500))
cv.imshow('Thres', img)
cv.waitKey(0)