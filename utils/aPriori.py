from __future__ import print_function
from __future__ import division
import cv2 as cv
import numpy as np
import random as rng
import math
import os

############################################################################################

def _find_information(image, count):
    list_area = []
    list_lung = []
    sum_area = 0
    sum_perimeter = 0

    _, contours, _ = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        print("Finding countours in image n." + str(count) + " , Contours n." + str(i+1))
        # Ignore isolated pixels or points that do not belong to the masses
        if(cv.contourArea(contours[i])>50 and cv.arcLength(contours[i], True)> 40):
            list_area.append(cv.contourArea(contours[i]))
            list_lung.append(cv.arcLength(contours[i], True))
            sum_area += cv.contourArea(contours[i])
            sum_perimeter += cv.arcLength(contours[i], True)

    return list_area, list_lung, sum_area, sum_perimeter
        
############################################################################################

def extract_information(ground_path):
    ground_images = os.listdir(ground_path)
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
        area , perimeter, sum_Area, sum_Lung = _find_information(img, count)
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

    return min_area,average_area,max_area,min_perimeter,average_perimeter,max_perimeter