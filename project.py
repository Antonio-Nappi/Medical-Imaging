'''
Our pipeline is based on the following main steps:
    - First: an SVM classifier is trained thanks to the haralick texture features extracted from the dataset.
            The dataset was first balanced and then agumented in order to well fit the classifier. So, it is divided in
            2 parts: n.582 mass images; n.586 no-mass images.
    - Second: Output images of the classifier are preprocessed in order to enphasize internal structures such as masses.
    - Third: U-Net Neural Network, whose input is the predicted output of the SVM classifier that was pre-processed in
            the previous step. The images are then analyzed in order to extract masses.
    - Fourth: Image Segmentation.
'''

from utils import data_preprocessing
from SVM_Classifier import SVM_Classifier
import cv2 as cv

nomass_path = "dataset\images\\nomass"
mass_path = "dataset\images\\mass"
overlay_path = "dataset\overlay"
test_path = "dataset\\test"
mask_path = "dataset\masks"
ground_path = "dataset\groundtruth"

# STEP 1:   We extract the features from the training set in order to fit the SVM classifier. This step ends with a list of
#           predicted masses (it is also shown the accuracy of the classifier).
classifier = SVM_Classifier(nomass_path, mass_path, overlay_path, mask_path, ground_path, test_path)
classifier.labelling()
classifier.extract_features()
classifier.train_classifier()
predicted_mass = classifier.prediction()

#STEP 2:    Pre-processing of the images to enhance internal structures, before to give them to the Neural Net.
#data_preprocessing.preprocessing(test_path, predicted_mass)

#STEP 3:





