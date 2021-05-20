import cv2
import os
import pyrebase
import pandas as pd
import numpy as np
import lungs_finder as lf

from pathlib import Path
from tqdm.notebook import tqdm
from joblib import Parallel, delayed

from helpers import process_data
from firebase_env import firebase_config


dataset = pd.read_csv("data_image.csv")

raw = "raw"
transformed = "transformed"

path_train_normal = os.path.join("train", "NORMAL")
path_train_pneu = os.path.join("train", "PNEUMONIA")

path_test_normal = os.path.join("test", "NORMAL")
path_test_pneu = os.path.join("test", "PNEUMONIA")

path_val_normal = os.path.join("val", "NORMAL")
path_val_pneu = os.path.join("val", "PNEUMONIA")


Path(os.path.join(transformed, path_train_normal)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(transformed, path_train_pneu)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(transformed, path_test_normal)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(transformed, path_test_pneu)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(transformed, path_val_normal)).mkdir(parents=True, exist_ok=True)
Path(os.path.join(transformed, path_val_pneu)).mkdir(parents=True, exist_ok=True)

IMAGE_RESOLUTION = (250, 250, 1)
BORDER = 30
print("Image shape: {}".format(IMAGE_RESOLUTION))

images_transformed = Parallel(n_jobs=4)(
    delayed(process_data)(img_path, IMAGE_RESOLUTION, BORDER)
    for img_path in dataset.full_path_raw
)

results = Parallel(n_jobs=4)(
    delayed(cv2.imwrite)(img_path_transformed, images_transformed[index] * 255)
    for index, img_path_transformed in enumerate(dataset.full_path_transformed)
)

if sum(results) == len(dataset.full_path_raw):
    print("All images were transformed!")

firebase = pyrebase.initialize_app(firebase_config)
storage = firebase.storage()


def save_image_firebase(image_path, storage):
    try:
        storage.child(image_path).put(image_path)
        return True
    except:
        return False


for img_path_transformed in dataset.full_path_transformed:
    # To linux
    aux = img_path_transformed.replace("\\", "/")
    save_image_firebase(aux, storage)
