import cv2 as cv
import numpy as np
import random as rng
from PIL import  Image
import math
import os
import copy

def nothing(immmagine):
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

def preprocess(src_gray):
    zeroToOne_img = normalize(src_gray)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(src_gray)

    mu = np.average(src_gray)
    tmp = normalize(img_clahe)
    img3 = np.zeros((tmp.shape[0], tmp.shape[1]))
    for i in range(img_clahe.shape[0]):
        for j in range(img_clahe.shape[1]):
            img3[i, j] = tmp[i, j] * (1 - math.exp(-(zeroToOne_img[i, j] / mu)))
    img3 = normalize(img3)
    m2 = regional_mean(img3, [16, 16])
    img5 = normalize(img3 + regional_mean(m2, [16, 16]))

    return img5

############################

mass_path = "dataset\\test\\"
mass_images = os.listdir(mass_path)

i = 1
for mass in mass_images:
    img = cv.imread(mass_path + mass, cv.IMREAD_GRAYSCALE)
    print("Processing image n." + str(i) + " ...")
    prep = preprocess(img)
    pil_img = Image.fromarray(prep) #convert to PIL Image
    print("Saving image n." + str(i) + " ...")
    pil_img.save("dataset\enhanced\\" + mass)
    i +=1


'''
img = cv.imread("dataset\images\mass\\24055355_1e10aef17c9fe149_MG_L_CC_ANON.tif", cv.IMREAD_GRAYSCALE)
prepr_img = preprocess(img)
pixels = cv.countNonZero(prepr_img)
tmp = copy.deepcopy(prepr_img)
print("Max value di PREP: %8.55f" %(np.amax(prepr_img)))
print("Min value di PREP: %8.55f" %(np.amin(prepr_img)))
print("Average value di PREP: %8.55f" %(np.average(prepr_img)))
print("Pixel not zero: ", pixels)


cv.namedWindow("th")
cv.createTrackbar("TH", "th", 100, 600, nothing)

while(1): #0-NERO   1-BIANCO #0.3981747604413089458
    prefix = 0.0001747604413089458
    th = (cv.getTrackbarPos("TH", "th"))+100
    val = (th/1000)+prefix
    print("val: %8.55f" %(val))
    ret, th_img = cv.threshold(tmp, val, 1, cv.THRESH_BINARY)
    pix2= cv.countNonZero(th_img)

    #_____________ STAMPE ________ #
    print("-------------------------------")
    print("Max value di th: %8.55f" %(np.amax(th_img)))
    print("Min value di th: %8.55f" %(np.amin(th_img)))
    print("Pixel not zero: ", pix2)
    print("-------------------------------")
    #_______________________________#

    cv.imshow("th", np.hstack((cv.resize(prepr_img, (600,500)), cv.resize(th_img, (600,500)))))

    key = cv.waitKey(1)
    if key == 27:
        break

cv.destroyAllWindows()
'''