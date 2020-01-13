from keras.models import load_model
import cv2 as cv
import numpy as np


class UNet:

    def __init__(self):
        self.unet = load_model("file\Model.h5")

    def unet_predict(self,predicted_mass):
        print("-------------------- [STATUS] Loading the U-Net model ----------------")
        unet_input = []
        print("-------------------- [STATUS] U-Net prediction of masses -------------")
        for mass in predicted_mass:
            img = cv.resize(mass,(512,512))
            img = np.reshape(img, (512, 512, 1))
            unet_input.append(img)
        predictions = self.unet.predict(np.asarray(unet_input))
        print("-------------------- [NOTIFY] All images have been processed ---------")
        return predictions

