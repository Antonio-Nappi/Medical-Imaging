from utils import data_preprocessing
from utils.jaccard import jaccard_function
from predictions.SVM_Classifier import SVM_Classifier
from predictions.UNet import UNet
from extraction import draw_masses

import os
import cv2 as cv
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
############################ PATH DEFINITION ############################
nomass_path = "dataset\images\\nomass"
mass_path = "dataset\images\\mass"
overlay_path = "dataset\overlay"
test_path = "dataset\\test"
mask_path = "dataset\masks"
ground_path = "dataset\groundtruth\\"
################################   END   ################################
'''
# STEP 1:   Extracting the features from the training set in order to fit the SVM classifier. This step ends with a list of
#           predicted masses (it is also shown the accuracy of the classifier).
classifier = SVM_Classifier(nomass_path, mass_path, overlay_path, mask_path, ground_path, test_path)
classifier.labelling()
classifier.extract_features()
classifier.train_classifier()
predicted_mass, path_predicted_mass = classifier.prediction()

#STEP 2:    Pre-processing of the images to enhance internal structures, before to give them to the Neural Net.
predicted_mass = data_preprocessing.preprocessing(predicted_mass)
predicted_mass = data_preprocessing.cropping(mask_path, predicted_mass, path_predicted_mass)
'''
#STEP 3:    Loading the U-Net model and predicting masses of test set
predicted_mass = []
path_predicted_mass=os.listdir("dataset/unet_input")
for path in os.listdir("dataset/unet_input"):
    img = cv.imread("dataset/unet_input/"+path,cv.IMREAD_ANYDEPTH)
    predicted_mass.append(img)

unet = UNet()
predictions = unet.unet_predict(predicted_mass)

scores = unet.evaluate(predicted_mass,path_predicted_mass)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
print("lunghezza: ", len(predictions))
print(predictions[0])
for prediction,path in zip(predictions,path_predicted_mass):
    cv.imwrite("dataset/predictions/"+path,prediction)
#STEP 4:    Segmentation process and final output
segmented_images = draw_masses.clean_unet_images(predicted_mass, predictions)
outcomes, ground_images = draw_masses.my_draw_contours(segmented_images)


#STEP 5:    Evaluating performance

jaccard_list,media=jaccard_function(ground_images,path_predicted_mass)