from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import random as rng
import math
import os

############################################################################################

def aPriori(image, count):
    list_area = []
    list_lung = []
    sum_area = 0
    sum_perimeter = 0

    _, contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        print("Finding countours in image n." + str(count) + " , Contours n." + str(i+1))
        #L'if è stato utilizzato per evitare i pixel isolati o piccolissime regioni di pixel che non fanno parte della massa
        if(cv.contourArea(contours[i])>50 and cv.arcLength(contours[i], True)> 40):
            list_area.append(cv.contourArea(contours[i]))
            list_lung.append(cv.arcLength(contours[i], True))
            sum_area += cv.contourArea(contours[i])
            sum_perimeter += cv.arcLength(contours[i], True)

    return list_area, list_lung, sum_area, sum_perimeter
        
############################################################################################

ground_path = "dataset\groundtruth"
ground_images = os.listdir("dataset\groundtruth")

list_areas = []
list_perimeters = []
sum_area_tot = 0
sum_perimeter_tot = 0

count = 1
for ground in ground_images:
    img = cv.imread(ground_path + "\\" + ground, cv.IMREAD_GRAYSCALE)
    if img is None:
        print('Could not open or find the image:', img)
        exit(0)
    area , perimeter, sum_Area, sum_Lung = aPriori(img, count)
    sum_area_tot += sum_Area
    sum_perimeter_tot += sum_Lung

    list_areas.extend(area)
    list_perimeters.extend(perimeter)
    count += 1
    
average_area = sum_area_tot/len(list_areas)
average_perimeter = sum_perimeter_tot/len(list_perimeters)

min_area = min(list_areas)
min_perimeter = min(list_perimeters)

max_area = max(list_areas)
max_perimeter = max(list_perimeters)
   
for i in range(len(list_area_tot)):
  print(' * Area[%d] = %.2f, Lung[%d] = %.2f' % (i, list_area_tot[i],i, list_lung_tot[i]))

print("Minimum Area: ", min_area)
print("Average Area: ", average_area)
print("Maximum Area: ", max_area)

print("Minimum Perimeter: ", min_perimeter)
print("Average Perimeter: ", average_perimeter)
print("Maximum Perimeter: ", max_perimeter)


''' 
______________OUTPUT______________
Valore Minimo per l'Area :  2995.5
Valore Medio per l'Area :  97040.43534482758
Valore Massimo per l'Area :  751132.0

Valore Minimo per la Lunghezza :  216.46803629398346
Valore Medio per la Lunghezza :  1303.7378783071863
Valore Massimo per la Lunghezza :  3855.347655892372
'''
