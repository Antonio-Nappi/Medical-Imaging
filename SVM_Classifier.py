import cv2 as cv
import os
import mahotas as mt
import sklearn.svm as skl
from sklearn.metrics import accuracy_score
from utils import dataset_preprocessing as dsp
from utils import utilities as ut


class SVM_Classifier:

    def __init__(self,nomass_path,mass_path, overlay_path, mask_path,ground_path):
        # Create the classifier
        print("--------- [STATUS] Creating the classifier ---------")
        self._my_svm = skl.SVC(C=100000, kernel='rbf', gamma='auto', verbose=True, max_iter=-1)
        self._nomass_images = os.listdir(nomass_path)
        self._mass_images = os.listdir(mass_path)
        # load the overlay dataset
        self._overlay_images = os.listdir(overlay_path)
        # load the mask dataset
        self._mask_images = os.listdir(mask_path)
        # load the groundtruth
        self._ground_images = os.listdir(ground_path)
        # empty list to hold feature vectors and train labels
        self._train_features = []
        self._train_labels = []

    def _texture_features(self,image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)
        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean

    def _evaluate_prediction(self,y_pred, y_true):
        predicted_mass = 0
        predicted_nomass = 0
        for y in y_pred:
            if y == 0:
                predicted_nomass +=1
            else:
                predicted_mass +=1
        print("Number of masses: 45")
        print("Number of predicted masses: ", predicted_mass)
        print("---------------------------------------")
        print("Number of non-masses: 60")
        print("Number of predicted non-masses: ", predicted_nomass)
        print("---------------------------------------")
        print("SVM ACCURACY: ", accuracy_score(y_true, y_pred))

    # load the training dataset

    def labelling(self,mass_path=None,nomass_path=None,test_path=None,labelling=False):
        if(labelling):
            dsp.cleaning(self._mask_images, self._nomass_images)
            dsp.createClasses(self._nomass_path, self._nomass_images, self._overlay_images, mass_path)
            dsp.mirroring(mass_path, self._overlay_images, self._ground_images)
            dsp.flipping(self._mass_images, self._nomass_images, self._ground_images)
            dsp.createTestSet(mass_path, nomass_path, test_path)

    def extract_features(self):
        print ("--------- [STATUS]: Started extracting haralick textures ---------")
        count_training = 1
        if not ut.check_file():
            for mass in self._mass_images:
                # read the training image
                image = cv.imread("dataset\images\mass\\" + mass, cv.IMREAD_GRAYSCALE)
                self._train_labels.append(int(1))
                # extract haralick texture from the image
                features = self._texture_features(image)
                # append the feature vector and label
                self._train_features.append(features)
                print("Extracting features from image number " + str(count_training))
                count_training += 1
            for nomass in self._nomass_images:
                image = cv.imread("dataset\images\\nomass\\" + nomass, cv.IMREAD_GRAYSCALE)
                self._train_labels.append(int(0))
                # extract haralick texture from the image
                features = self._texture_features(image)
                # append the feature vector and label
                self._train_features.append(features)
                print("Extracting features from image number " + str(count_training))
                count_training += 1
            ut.store(self._train_features,self._train_labels)
        else:
            self._train_features,self._train_labels = ut.load()

    def train_classifier(self):
        # Fit the training data and labels
        print ("--------- [STATUS]: Fitting data ---------")
        self._my_svm.fit(self._train_features, self._train_labels)


    def prediction(self,test_path,tot_test_images=105,num_mass_test=45):
        # load the test set
        test_images = os.listdir(test_path)

        # Loop over the test images
        count_test = 1
        y_pred = []
        y_true = []
        predicted_mass = []
        for test in test_images:
            image = cv.imread(test_path + "\\" + test)
            # convert to grayscale
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            # extract haralick texture from the image
            features = self._texture_features(gray)

            # evaluate the model and predict label
            prediction = int(self._my_svm.predict(features.reshape(1, -1))[0])
            if prediction:
                predicted_mass.append(test)
            print("------------------------------")
            print("Prediction of image number ", str(count_test), ": ", prediction)
            y_pred.append(prediction)

            # show the label : 0 means noMass, while 1 means mass
            count_test += 1

        # Evaluate SVM prediction
        for i in range(tot_test_images):
            if i < num_mass_test:
                y_true.append(int(1))
            else:
                y_true.append(int(0))

        self._evaluate_prediction(y_pred, y_true)
        return predicted_mass
