import cv2 as cv
import numpy as np
import os
import shutil
from PIL import Image
import mahotas as mt
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import utils

nomass_path = "dataset\images\\nomass"
nomass_images = os.listdir(nomass_path)
mass_path = "dataset\images\mass"

# load the overlay dataset
overlay_path = "dataset\overlay"
overlay_images = os.listdir(overlay_path)

# load the mask dataset
mask_path = "dataset\masks"
mask_images = os.listdir(mask_path)

labelling = False #set True if you want to create class labels
if(labelling):
    utils.clearing(mask_images, nomass_images)
    utils.createClasses(nomass_path, nomass_images, overlay_images, mass_path)
    utils.mirroring(mass_path)

# load the training dataset
mass_images = os.listdir(mass_path)
images = [None]*30
out=[]

for i in range(30):
    print(mass_images[i])
    string = mass_path+'\\'+mass_images[i]
    images[i] = cv.imread(string,cv.IMREAD_GRAYSCALE)

k = 0
for img in images:
    tmp1 = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    tmp2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    low_bound = np.percentile(img,5)
    up_bound = np.percentile(img,95)
    max = np.amax(img)
    min = np.amin(img)
    print("Immagine numero {}".format(k))
    k += 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp1[i,j] = (img[i,j]) * ((up_bound - low_bound)/(max - min)) + low_bound
            tmp2[i, j] =255-(255 / math.log(928, 10)) * math.log(10 + 9 * tmp1[i, j], 10)
    tmp2=cv.resize(tmp2,(700,700))
    cv.imshow("Test_Image", tmp2)
    cv.waitKey(0)