import cv2 as cv
import numpy as np
import os
from PIL import Image

mass_path = "dataset\images\mass"
mask_path = "dataset\masks"
gt_path = "dataset\groundtruth"
mass_images = os.listdir(mass_path)
mask_images = os.listdir(mask_path)
print(len(mask_images))
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
    gt_img=cv.imread(gt_path+"\\" + entry[:-9] + ".tif",cv.IMREAD_GRAYSCALE)
    gt_image = Image.fromarray(gt_img)
    gt_cropped = gt_image.crop((x,y,x+width,y+height))
    image = Image.fromarray(img_mass)
    cropped = image.crop((x, y, x+width, y+height))
    cropped.save("dataset\images\cropped\\" + entry)
    gt_cropped.save("dataset\groundtruth\cropped\\"+entry)

