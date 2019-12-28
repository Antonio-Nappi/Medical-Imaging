import os
import numpy as np

def check_file():
    return os.path.exists("..\Features.txt") and os.path.exists("..\Labels.txt")

def load():
    with open("..\Features.txt","r") as features_file:
        train_features = np.loadtxt(features_file)
    with open("..\Labels.txt","r") as labels_file:
        train_labels = np.loadtxt(labels_file)
    return train_features,train_labels

def store(train_features,train_labels):
    with open("..\Features.txt", "w") as features_file:
        np.savetxt(features_file,train_features)
    with open("..\Labels.txt", "w") as labels_file:
        np.savetxt(labels_file,train_labels)