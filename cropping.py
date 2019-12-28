import cv2 as cv
import numpy as np
import os
from PIL import Image
import math

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


mass_path = "dataset\images\mass"
mask_path = "dataset\masks"
gt_path = "dataset\groundtruth"
mass_images = os.listdir(mass_path)
mask_images = os.listdir(mask_path)
i = 0
for entry in mask_images:
    i+=1
    print(i)
    # Find rectangles
    mask = cv.imread(mask_path + "\\" + entry, cv.IMREAD_GRAYSCALE)
    retval, labels = cv.connectedComponents(mask, ltype=cv.CV_16U)
    labels = np.asarray(labels, np.uint8)
    contours, hierarchy = cv.findContours(labels, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    x, y, width, height = cv.boundingRect(contours[0])
    # Cropping images
    print(mass_path + "\\" + entry[:-9] + ".tif")
    img_mass = cv.imread(mass_path + "\\" + entry[:-9] + ".tif", cv.IMREAD_GRAYSCALE)
    if img_mass is None:
        continue
    image = Image.fromarray(img_mass)
    cropped = image.crop((x, y, x+width, y+height))
    cropped.save("dataset\images\cropped\\" +  entry[:-9] + ".tif")
    cropped=cv.imread(mask_path + "\\" + entry, cv.IMREAD_GRAYSCALE)
    cv.imwrite("dataset\images\cropped\enhanced\\"+entry[:-9] + ".tif",preprocess(cropped))

