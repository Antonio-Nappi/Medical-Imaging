import cv2 as cv
import numpy as np
import os
import shutil
from PIL import Image
import mahotas as mt

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