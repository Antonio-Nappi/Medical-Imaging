import numpy as np
import math
import cv2 as cv
from PIL import Image

##########################################################################################
def __normalize(img):
    tmp = img-np.amin(img)
    image = tmp/np.amax(img)
    return image

def __regional_mean(img,list):
    tmp = cv.blur(img,(list[0],list[1]))
    return cv.resize(tmp, (img.shape[0],img.shape[1]),interpolation=cv.INTER_LINEAR)

def __normalize_and_HE(img):
    t = cv.equalizeHist(np.asarray(__normalize(img),dtype=np.uint8))
    return t

def __enhancing_structures(src_gray):
    zeroToOne_img = __normalize(src_gray)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(src_gray)

    mu = np.average(src_gray)
    tmp = __normalize(img_clahe)
    img3 = np.zeros((tmp.shape[0], tmp.shape[1]))
    for i in range(img_clahe.shape[0]):
        for j in range(img_clahe.shape[1]):
            img3[i, j] = tmp[i, j] * (1 - math.exp(-(zeroToOne_img[i, j] / mu)))
    img3 = __normalize(img3)
    m2 = __regional_mean(img3, [16, 16])
    img5 = __normalize(img3 + __regional_mean(m2, [16, 16]))

    return img5
##########################################################################################

def preprocessing(predicted_mass):
    print("--------- [STATUS] Preprocessing images ---------")
    i = 1
    enhanced_mass = []
    for mass in predicted_mass:
        print("Processing image n." + str(i) + " ...")
        mass = cv.cvtColor(mass, cv.COLOR_BGR2GRAY)
        prep_img = __enhancing_structures(mass)
        enhanced_mass.append(prep_img)
        i +=1
    print("-------------------- [NOTIFY] Image preprecessed ---------------------")
    return enhanced_mass

##########################################################################################
def build_true_path(path):
    path = path.split("_")
    path.pop(0)
    true_path = ""
    for p in path:
        true_path += p + "_"

    return true_path[:-4] + "mask.png"
##########################################################################################

def cropping(mask_path, mass_images, path_predicted_mass):
    print("-------------------- [STATUS] Cropping images for U-Net --------------")
    cropped_images = []
    i = 0
    for p in path_predicted_mass:
        # Find rectangles
        path = build_true_path(p)
        mask = cv.imread(mask_path + "\\" + path, cv.IMREAD_GRAYSCALE)
        retval, labels = cv.connectedComponents(mask, ltype=cv.CV_16U)
        labels = np.asarray(labels, np.uint8)
        contours, hierarchy = cv.findContours(labels, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        x, y, width, height = cv.boundingRect(contours[0])

        # Cropping images
        print("Cropping image n.", i+1)
        image = Image.fromarray(mass_images[i])
        img_cropped = image.crop(box=(x, y, x+width, y+height))
        cropped_images.append(img_cropped)

        # Saving images
        #pil_img = Image.fromarray(img_cropped)  # convert to PIL Image
        print("Saving image n.", i+1)
        print("------------------------------")
        img_cropped.save("dataset\\unet_input\\" + path_predicted_mass[i])
        i += 1

    print("-------------------- [NOTIFY] All images have been cropped -----------")
    return cropped_images
