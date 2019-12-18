import cv2 as cv
import numpy as np
import os
import shutil
from PIL import Image
import mahotas as mt
import math
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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

# load the overlay dataset
overlay_path = "dataset\overlay"
overlay_images = os.listdir(overlay_path)

# load the mask dataset
mask_path = "dataset\masks"
mask_images = os.listdir(mask_path)

labelling = False #set True if you want to create class labels
if(labelling):
    clearing(mask_images, nomass_images)
    createClasses(nomass_path, nomass_images, overlay_images, mass_path)
    mirroring(mass_path)

# load the training dataset
mass_images = os.listdir(mass_path)
images = [None]*30
out=[]
#####CODICE DI PROVA#########
for i in range(30):
    print(mass_images[i])
    string = mass_path+'\\'+mass_images[i]
    images[i] = cv.imread(string,cv.IMREAD_GRAYSCALE)

k = 0
for img in images:
    tmp1 = np.zeros((img.shape[0],img.shape[1]),np.uint8)
    tmp2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    low_bound = np.percentile(img,5)
    up_bound = np.percentile(img,95)
    max = np.amax(img)
    min = np.amin(img)
    print("Immagine numero {}".format(k))
    k += 1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            tmp1[i,j] = (img[i,j]) * ((up_bound - low_bound)/(max - min)) + low_bound
            tmp2[i, j] =255-(255 / math.log(928, 10)) * math.log(10 + 9 * tmp1[i, j], 10)
    tmp2=cv.resize(tmp2,(700,700))
    cv.imshow("Test_Image", tmp2)
    cv.waitKey(0)
######FINE CODICE DI PROVA#######

'''
# empty list to hold feature vectors and train labels
train_features = []
train_labels = []

# Dividing test set from training set. The test set is made up of: 10% samples with mass and 10% samples without mass.
test_images = []
training_images = []

ten_percent_nomass = nomass_images[273:]
ten_percent_mass =  mass_images[194:]

test_images.extend(ten_percent_mass)
test_images.extend(ten_percent_nomass)

nomass_images = nomass_images[:273]
mass_images = mass_images[:194]

training_images.extend(nomass_images)
training_images.extend(mass_images)


# Extracting features from the training samples by using Haralick - GLCM
#convention 1=nomass; 0=mass
print ("--------- [STATUS]: Started extracting haralick textures... ---------")
count_training = 1
if not check_file():
    for train_image in training_images:
        if count_training <= 273:
            # read the training image
            image = cv.imread("dataset\images\\nomass\\" + train_image,cv.IMREAD_GRAYSCALE)
            train_labels.append(1)
        else:
            image = cv.imread("dataset\images\\mass\\" + train_image,cv.IMREAD_GRAYSCALE)
            train_labels.append(0)

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
clf_svm = SVC(random_state=9,max_iter=-1,verbose=True,)
#svm = LinearSVC(C=10.0,dual=False)
k = 10 # for k-fold validation
# Parameters and values to be tried
#params={'C': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]}
#grid_search=GridSearchCV(svm, params, cv=k)

# Fit the training data and labels
print ("--------- [STATUS]: Fitting data/label to model... ---------")
#grid_search.fit(train_features, train_labels)
clf_svm.fit(train_features, train_labels)
count_test = 1
# Loop over the test images
for test in test_images:
    if count_test <= 20:
        # read the test image
        image = cv.imread("dataset\images\\mass\\" + test)
    else:
        image = cv.imread("dataset\images\\nomass\\" + test)

    # convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # extract haralick texture from the image
    features = extract_features(gray)

    # evaluate the model and predict label
    prediction = clf_svm.predict(features.reshape(1, -1))[0]
    if prediction==1:
        prediction = 'nomass'
    else:
        prediction = 'mass'
    print("------------------------------")
    print("Prediction of image number ", str(count_test), ": ", prediction)
    print("-------------------------------")

    # show the label

    cv.putText(image, prediction, (20,30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    count_test += 1

    # display the output image
    cv.imshow("Test_Image", image)
    cv.waitKey(0)
'''

