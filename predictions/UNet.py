from keras.models import load_model
import cv2 as cv
import numpy as np

def unet_predict(predicted_mass):
    print("-------------------- [STATUS] Loading the U-Net model ----------------")
    model = load_model("file\Model.h5")
    unet_input = []
    print("-------------------- [STATUS] U-Net prediction of masses -------------")
    for mass in predicted_mass:
        mass = cv.resize(mass, (512, 512))
        mass = np.reshape(mass, (512, 512, 1))
        unet_input.append(mass)
    predictions = model.predict(np.asarray(unet_input))
    print("-------------------- [NOTIFY] All images have been processed ---------")
    return predictions