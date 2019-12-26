import cv2 as cv
import numpy as np
import os
import shutil
from PIL import Image
import mahotas as mt
import sklearn.svm as skl

#####################################__UTILS__#######################################
# data augmentation
def mirroring(path):
    mass_images = os.listdir(path)
    for image in mass_images:
        img = Image.open(path + '\\' + image)
        rotated_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        rotated_image.save(path + '\\' + 'rotated' + image)

# create labelled classes
def createClasses(source_path, all_images, overlay_images, dest_path):
    mass_images = []
    num_mass = 1
    for image in all_images:
        if image in overlay_images:
            mass_images.append(image)
            shutil.move(source_path + '\\' + image, dest_path) #it also remove the file from the source directory
            print("Processing image number: " + str(num_mass))
            num_mass += 1

def clearing(mask_images, all_images):
    count = 1
    for image in all_images:
        img_name = image[:-4]
        for mask in mask_images:
            if img_name in mask:
                img = cv.imread("dataset\images\\nomass" + '\\' + image)
                cutter = cv.imread("dataset\masks" + '\\' + mask)
                img[cutter == 0] = 0 #apply the mask on the image in order to clean the background
                print("Cleaning image number " + str(count))
                count += 1
                break

def flipping(mass_images, nomass_images):
    for image in mass_images:
        img = Image.open("dataset\images\mass\\" + image)
        rotated = img.rotate(45)
        rotated.save("dataset\images\mass\ROT_" + image)
    for image in nomass_images:
        img = Image.open("dataset\images\\nomass\\" + image)
        rotated = img.rotate(45)
        rotated.save("dataset\images\\nomass\ROT_" + image)

def extract_features(image):
    # calculate haralick texture features for 4 types of adjacency
    textures = mt.features.haralick(image)

    # take the mean of it and return it
    ht_mean = textures.mean(axis=0)
    return ht_mean

def check_file():
    return os.path.exists("Features.txt") and os.path.exists("Labels.txt")

def load():
    with open("Features.txt","r") as features_file:
        train_features = np.loadtxt(features_file)
    with open("Labels.txt","r") as labels_file:
        train_labels = np.loadtxt(labels_file)
    return train_features,train_labels

def store(train_features,train_labels):
    with open("Features.txt", "w") as features_file:
        np.savetxt(features_file,train_features)
    with open("Labels.txt", "w") as labels_file:
        np.savetxt(labels_file,train_labels)
################################__END UTILS__##########################################

# load the training dataset
nomass_path = "dataset\images\\nomass"
nomass_images = os.listdir(nomass_path)
mass_path = "dataset\images\mass"
mass_images = os.listdir(mass_path)

# load the overlay dataset
overlay_path = "dataset\overlay"
overlay_images = os.listdir(overlay_path)

# load the mask dataset
mask_path = "dataset\masks"
mask_images = os.listdir(mask_path)

# load the test set
test_path = "dataset\\test"
test_images = os.listdir(test_path)

labelling = False #set True if you want to create class labels
if(labelling):
    clearing(mask_images, nomass_images)
    createClasses(nomass_path, nomass_images, overlay_images, mass_path)
    mirroring(mass_path)
    flipping(mass_images, nomass_images)

# empty list to hold feature vectors and train labels
train_features = []
train_labels = []

# Extracting features from the training samples by using Haralick - GLCM
print ("--------- [STATUS]: Started extracting haralick textures... ---------")
count_training = 1
if not check_file():
    for mass in mass_images:
        # read the training image
        image = cv.imread("dataset\images\mass\\" + mass, cv.IMREAD_GRAYSCALE)
        train_labels.append(0)
        # extract haralick texture from the image
        features = extract_features(image)
        # append the feature vector and label
        train_features.append(features)

        print("Extracting features from image number " + str(count_training))
        count_training += 1
    for nomass in nomass_images:
        image = cv.imread("dataset\images\\nomass\\" + nomass, cv.IMREAD_GRAYSCALE)
        train_labels.append(1)
        # extract haralick texture from the image
        features = extract_features(image)
        # append the feature vector and label
        train_features.append(features)

        print("Extracting features from image number " + str(count_training))
        count_training += 1
    #store(train_features,train_labels)
    store(train_features,train_labels)
else:
    train_features,train_labels = load()

# Create the classifier
print ("--------- [STATUS] Creating the classifier... ---------")
clf_svm = skl.SVC(C=1.5, kernel='rbf', gamma='auto', verbose=True, max_iter=-1)

# Fit the training data and labels
print ("--------- [STATUS]: Fitting data/label to model... ---------")
clf_svm.fit(train_features, train_labels)

test_path = "dataset\\test"
test_images = os.listdir(test_path)

count_test = 1
# Loop over the test images
for test in test_images:
    image = cv.imread("dataset\\test\\" + test)
    # convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # extract haralick texture from the image
    features = extract_features(gray)

    # evaluate the model and predict label
    prediction = clf_svm.predict(features.reshape(1, -1))[0]
    print("------------------------------")
    # 1 means noMass, while 0 means mass
    print("Prediction of image number ", str(count_test), ": ", prediction)

    # show the label
    count_test += 1


