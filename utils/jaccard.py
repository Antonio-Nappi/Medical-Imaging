import cv2 as cv
import numpy as np
from utils.data_preprocessing import build_true_path

############################INNER FUNCTION##################################

def __jaccard_similarity(im1, im2):
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    return intersection.sum() / float(union.sum())

#############################################################################

    
def jaccard(ground_images, path_images):
    print("-------------------- [STATUS] Computing Jaccard index ----------------")
    true_path = "dataset\groundtruth\ground_test"
    
    i = 0
    somma = 0
    jaccard_list = []
    for i in range(len(ground_images)):
      
        path = build_true_path(path_images[i])

        true_img = cv.imread(true_path + "\\" + path, cv.IMREAD_ANYDEPTH)
        true_img = cv.resize(true_img,(512, 512)) 
           
        img_true = np.asarray(true_img).astype(np.bool)
        img_pred = np.asarray(ground_images[i]).astype(np.bool)
        
        jaccard = __jaccard_similarity(img_true, img_pred)
        jaccard_list.append(jaccard)
        somma = somma + jaccard

    media = somma/(i+1)
    
    return jaccard_list, media