import cv2 as cv
import numpy as np
import os
import shutil
from PIL import Image
import mahotas as mt
import sklearn.svm as skl
from sklearn.metrics import accuracy_score

#####################################__UTILS__#######################################
# data augmentation
def mirroring(mass_path, overlay_images, ground_images):
    n_mass = 1
    n_overlay = 1
    n_ground = 1
    mass_images = os.listdir(mass_path)
    for image in mass_images:
        print("Mirroring mass image n.", n_mass)
        img = Image.open(mass_path + '\\' + image)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_image.save(mass_path + '\\' + 'mirror' + image)
        n_mass += 1
    for image in overlay_images:
        print("Mirroring overlay image n.", n_overlay)
        img = Image.open('dataset\overlay' + '\\' + image)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_image.save('dataset\overlay' + '\\' + 'mirror' + image)
        n_overlay += 1
    for image in ground_images:
        print("Mirroring ground image n.", n_ground)
        img = Image.open('dataset\groundtruth' + '\\' + image)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_image.save('dataset\groundtruth' + '\\' + 'mirror' + image)
        n_ground += 1

def flipping(mass_images, nomass_images, ground_images):
    n_mass = 1
    n_nomass = 1
    n_ground = 1
    for image in mass_images:
        print("Flipping mass image n.", n_mass)
        img = Image.open("dataset\images\mass\\" + image)
        rotated = img.rotate(45)
        rotated.save("dataset\images\mass\ROT_" + image)
        n_mass += 1
    for image in nomass_images:
        print("Flipping nomass image n.", n_nomass)
        img = Image.open("dataset\images\\nomass\\" + image)
        rotated = img.rotate(45)
        rotated.save("dataset\images\\nomass\ROT_" + image)
        n_nomass += 1
    for image in ground_images:
        print("Flipping ground image n.", n_ground)
        img = Image.open("dataset\groundtruth\\" + image)
        rotated = img.rotate(45)
        rotated.save("dataset\groundtruth\ROT_" + image)
        n_ground += 1

# create labelled classes
def createClasses(source_path, all_images, overlay_images, dest_path):
    mass_images = []
    num_mass = 1
    for image in all_images:
        if image in overlay_images:
            mass_images.append(image)
            shutil.move(source_path + '\\' + image, dest_path) #it also remove the file from the source directory
            print("Processing image n. " + str(num_mass))
            num_mass += 1

def createTestSet(mass_path, nomass_path, test_path):
    mass_images = os.listdir(mass_path)
    #Takes 10% mass images for the test set
    for i in range(45):
        print("Moving image n.", i)
        shutil.move(mass_path + '\\' + mass_images[i], test_path)

    nomass_images = os.listdir(nomass_path)
    #Takes 10% mass images for the test set
    for i in range(60):
        print("Moving image n.", i)
        shutil.move(nomass_path + '\\' + nomass_images[i], test_path)
    '''
    RENAME FILES
    test_images = os.listdir(test_path)
    for image in test_images:
        os.rename(test_path + "\\" + image, test_path + "\\" + "NOMASS_" + image)
    '''

# removing background noise
def cleaning(mask_images, all_images):
    count = 1
    for image in all_images:
        img_name = image[:-4]
        for mask in mask_images:
            if img_name in mask:
                img = cv.imread("dataset\images\\nomass" + '\\' + image)
                cutter = cv.imread("dataset\masks" + '\\' + mask)
                img[cutter == 0] = 0 #apply the mask on the image in order to clean the background
                print("Cleaning image n. " + str(count))
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

def evaluate_prediction(y_pred, y_true):
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

################################__END UTILS__##########################################

# load the training dataset
nomass_path = "dataset\images\\nomass"
nomass_images = os.listdir(nomass_path)
mass_path = "dataset\images\mass"
mass_images = os.listdir(mass_path)

# load the overlay dataset
overlay_path = "dataset\overlay"
overlay_images = os.listdir(overlay_path)

# declare test set
test_path = "dataset\\test"

# load the mask dataset
mask_path = "dataset\masks"
mask_images = os.listdir(mask_path)

#load the groundtruth
ground_path = "dataset\groundtruth"
ground_images = os.listdir(ground_path)

labelling = False #set True if you want to create class labels
if(labelling):
    cleaning(mask_images, nomass_images)
    createClasses(nomass_path, nomass_images, overlay_images, mass_path)
    mirroring(mass_path, overlay_images, ground_images)
    flipping(mass_images, nomass_images, ground_images)
    createTestSet(mass_path, nomass_path, test_path)


# empty list to hold feature vectors and train labels
train_features = []
train_labels = []

# Extracting features from the training samples by using Haralick
print ("--------- [STATUS]: Started extracting haralick textures... ---------")
count_training = 1
if not check_file():
    for mass in mass_images:
        # read the training image
        image = cv.imread("dataset\images\mass\\" + mass, cv.IMREAD_GRAYSCALE)
        train_labels.append(int(1))
        # extract haralick texture from the image
        features = extract_features(image)
        # append the feature vector and label
        train_features.append(features)
        print("Extracting features from image number " + str(count_training))
        count_training += 1

    for nomass in nomass_images:
        image = cv.imread("dataset\images\\nomass\\" + nomass, cv.IMREAD_GRAYSCALE)
        train_labels.append(int(0))
        # extract haralick texture from the image
        features = extract_features(image)
        # append the feature vector and label
        train_features.append(features)
        print("Extracting features from image number " + str(count_training))
        count_training += 1

    store(train_features,train_labels)
else:
    train_features,train_labels = load()

# Create the classifier
print ("--------- [STATUS] Creating the classifier... ---------")
svm = skl.SVC(C=100000, kernel='rbf', gamma='auto', verbose=True, max_iter=-1)

# Fit the training data and labels
print ("--------- [STATUS]: Fitting data/label to model... ---------")
svm.fit(train_features, train_labels)

# load the test set
test_images = os.listdir(test_path)

# Loop over the test images
count_test = 1
y_pred = []
y_true = []
for test in test_images:
    image = cv.imread(test_path + "\\" + test)
    # convert to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # extract haralick texture from the image
    features = extract_features(gray)

    # evaluate the model and predict label
    prediction = int(svm.predict(features.reshape(1, -1))[0])
    print("------------------------------")
    print("Prediction of image number ", str(count_test), ": ", prediction)

    y_pred.append(prediction)

    # show the label : 0 means noMass, while 1 means mass
    count_test += 1

# Evaluate SVM prediction
for i in range(105):
    if i < 45:
        y_true.append(int(1))
    else:
        y_true.append(int(0))

evaluate_prediction(y_pred, y_true)
