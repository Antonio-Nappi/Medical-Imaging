import cv2 as cv
import numpy as np
import os
from PIL import Image

mass_path = "dataset\images\mass"
mask_path = "dataset\masks"
mass_images = os.listdir(mass_path)
mask_images = os.listdir(mask_path)

list = []

for entry in mask_images:
    # Find rectangles
    mask = cv.imread(mask_path + "\\" + entry, cv.IMREAD_GRAYSCALE)
    retval, labels = cv.connectedComponents(mask, ltype=cv.CV_16U)
    labels = np.asarray(labels, np.uint8)
    contours, hierarchy = cv.findContours(labels, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    x, y, width, height = cv.boundingRect(contours[0])

    # Cropping images
    img_mass = cv.imread(mass_path + "\\" + entry[:-4] + ".tif", cv.IMREAD_GRAYSCALE)
    image = Image.fromarray(img_mass)
    cropped = image.crop(x, y, x+width, y+height)
    cropped.save("dataset\images\cropped\\" + entry)

