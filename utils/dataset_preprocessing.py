import os
from PIL import Image
import cv2 as cv
import shutil

# data balancing for binary classification
def balancing(mass_path):
    n_mass = 1
    mass_images = os.listdir(mass_path)
    for image in mass_images:
        img = Image.open(mass_path + '\\' + image)
        print("Mirroring mass image n.", n_mass)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        print("Flipping mass image n.", n_mass)
        rotated = img.rotate(45, expand=True)
        rotated.save(mass_path + "\\" + "rot45_" + image)
        mirror_image.save(mass_path + '\\' + 'mirror_' + image)
        n_mass += 1

# data augmentation
def augmentation(mass_path, nomass_path, nomass_images):
    mass_images = os.listdir(mass_path)
    n_mass = 1
    n_nomass = 1
    for image in mass_images:
        print("Rotating mass image n.", n_mass)
        img = Image.open(mass_path + "\\" + image)
        rotated = img.rotate(315, expand=True)
        rotated.save(mass_path + "\\" + "rot315_" + image)
        n_mass += 1
    for image in nomass_images:
        print("Flipping nomass image n.", n_nomass)
        img = Image.open(nomass_path + "\\" + image)
        rotated = img.rotate(315, expand=True)
        rotated.save(nomass_path + "\\" + "rot315_" + image)
        n_nomass += 1

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

# removing background noise
def cleaning(mask_path, mask_images, all_path, all_images):
    count = 1
    for image in all_images:
        img_name = image[:-4]
        for mask in mask_images:
            if img_name in mask:
                img = cv.imread(all_path + '\\' + image)
                cutter = cv.imread(mask_path + '\\' + mask)
                img[cutter == 0] = 0 #apply the mask on the image in order to clean the background
                print("Cleaning image n. " + str(count))
                count += 1
                break
