'''
La nostra idea è quella di mettere in cascata un classificatore SVM (addestrato con le feature estratta da haralick) con
un processo (semiautomatico per adesso è quello che da i risultati migliori, se la unet funzionasse useremmo questa) in grado
di effettuare la segmentazione dell'immagine(mediante l'estrazione dei bordi) per determinati criteri. L'immagine viene
prima di essere segmentata, elaborata con una serie di trasformazioni suggerite dal paper che ci ha passato il prof.
'''

from utils import data_preprocessing
from SVM_Classifier import SVM_Classifier
import cv2 as cv

nomass_path = "dataset\images\\nomass\\"
mass_path = "dataset\images\\mass\\"
overlay_path = "dataset\overlay\\"
test_path = "dataset\\test\\"
mask_path = "dataset\masks\\"
ground_path = "dataset\groundtruth\\"

#Creato il classificatore ed ottenuta la cartella contenente solo le masse
classifier = SVM_Classifier(nomass_path,mass_path, overlay_path, mask_path,ground_path)
classifier.labelling(mass_path,nomass_path,test_path)
classifier.extract_features()
classifier.train_classifier()
predicted_mass = classifier.prediction(test_path)

#Fatto l'enhancement di tutte le immagini predette come masse
data_preprocessing.preprocessing(test_path, predicted_mass)
#a_min=input("Insert the minimum area that the mass should have(press -1 to use the precalculated area):")

#Elaborazione delle immagini





