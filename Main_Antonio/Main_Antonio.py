# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:44:01 2020

@author: AntonioBho
"""



from draw_mass import drawer
from utils.utilities import extract_information
 
import os
import cv2 as cv


############################ PATH DEFINITION ############################
nomass_path = "dataset/images/nomass"
mass_path = "dataset/images/mass"
overlay_path = "dataset/overlay"
test_path = "dataset/test"
mask_path = "dataset/masks"
ground_path = "dataset_Antonio/mask_true"
################################   END   ################################



predictions = []
predicted_mass = []
after_mask=[]
path_predicted_mass = os.listdir("dataset_Antonio/INPUT UNET")
path_predictions = os.listdir("dataset_Antonio/OUTPUT UNET")
path_aftermask = os.listdir("dataset_Antonio/AfterMask")




for p in path_predicted_mass[:-1]:
    path = "dataset_Antonio/INPUT UNET/" + p
    img = cv.imread(path, cv.IMREAD_ANYDEPTH)
    predicted_mass.append(img)

for p in path_predictions[:-1]:
    path = "dataset_Antonio/OUTPUT UNET/" + p
    img = cv.imread(path, cv.IMREAD_ANYDEPTH)
    predictions.append(img)
    
for p in path_aftermask[:-1]:
    path = "dataset_Antonio/AfterMask/" + p
    img = cv.imread(path, cv.IMREAD_ANYDEPTH)
    after_mask.append(img)



print("INPUT : ", len(predicted_mass))    
print("OUTPUT :",len(predictions))  
print("AFTERMASKDIRECTLY :",len(after_mask))
segmented_images = drawer.clean_unet_images(predicted_mass, predictions)
outcomes, ground_images = drawer.my_draw_contours(segmented_images, ground_path, path_predicted_mass)

