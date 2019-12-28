import os
from PIL import Image
import cv2 as cv
import shutil

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
        img = Image.open('..\dataset\overlay' + '\\' + image)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_image.save('..\dataset\overlay' + '\\' + 'mirror' + image)
        n_overlay += 1
    for image in ground_images:
        print("Mirroring ground image n.", n_ground)
        img = Image.open('..\dataset\groundtruth' + '\\' + image)
        mirror_image = img.transpose(Image.FLIP_LEFT_RIGHT)
        mirror_image.save('..\dataset\groundtruth' + '\\' + 'mirror' + image)
        n_ground += 1

def flipping(mass_images, nomass_images, ground_images):
    n_mass = 1
    n_nomass = 1
    n_ground = 1
    for image in mass_images:
        print("Flipping mass image n.", n_mass)
        img = Image.open("..\dataset\images\mass\\" + image)
        rotated = img.rotate(45)
        rotated.save("..\dataset\images\mass\ROT_" + image)
        n_mass += 1
    for image in nomass_images:
        print("Flipping nomass image n.", n_nomass)
        img = Image.open("..\dataset\images\\nomass\\" + image)
        rotated = img.rotate(45)
        rotated.save("..\dataset\images\\nomass\ROT_" + image)
        n_nomass += 1
    for image in ground_images:
        print("Flipping ground image n.", n_ground)
        img = Image.open("..\dataset\groundtruth\\" + image)
        rotated = img.rotate(45)
        rotated.save("..\dataset\groundtruth\ROT_" + image)
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

# removing background noise
def cleaning(mask_images, all_images):
    count = 1
    for image in all_images:
        img_name = image[:-4]
        for mask in mask_images:
            if img_name in mask:
                img = cv.imread("..\dataset\images\\nomass" + '\\' + image)
                cutter = cv.imread("..\dataset\masks" + '\\' + mask)
                img[cutter == 0] = 0 #apply the mask on the image in order to clean the background
                print("Cleaning image n. " + str(count))
                count += 1
                break
