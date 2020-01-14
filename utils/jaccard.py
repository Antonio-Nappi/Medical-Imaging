import cv2 as cv
import numpy as np
from utils.data_preprocessing import build_true_path

############################INNER FUNCTION##################################

def __jaccard_similarity(im_true, im_pred):
    '''
    The function computes the Jaccard index.
    :param im_true: the true groundtruth of the current image.
    :param im_pred: the predicted groundtruth of the current image.
    :return: the Jaccard index.
    '''
    if im_true.shape != im_pred.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im_true, im_pred)

    union = np.logical_or(im_true, im_pred)

    return intersection.sum() / float(union.sum())

#############################################################################

    
def jaccard(ground_images, path_images):
    '''
    The function computes the Jaccard indexes for all the predicted images.
    :param ground_images: the list of groundtruths of the predicted masses.
    :param path_images: the list of paths of the images.
    :return: a tuple of different elements:
        - jaccard_list: the list with all the Jaccard indexes;
        - average: the average of all the Jaccard indexes.
    '''
    print("-------------------- [STATUS] Computing Jaccard index ----------------")
    true_path = "dataset\groundtruth\ground_test"

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

    average = somma/(i+1)
    
    return jaccard_list, average