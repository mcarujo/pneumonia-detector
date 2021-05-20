import logging
import os

import cv2
import pyrebase
import lungs_finder as lf
import numpy as np
import pandas as pd
from pathlib import Path

from settings import database, firebase_config, token


class DataProcessing:
    def __init__(self):
        """
        Constructor setting some local variables.
        """
        self.IMAGE_RESOLUTION = (250, 250, 1)
        self.BORDER = 30

        if database == "FIREBASE":
            self.path_base = 'data'
            self.firebase = pyrebase.initialize_app(firebase_config)
            self.download_dataset()
        else:
            self.transformed_path = os.path.join(
                *("..", "data") if database == 'LOCAL' else "data")

    def load_datasets(self):
        """
        Load the dataset with the images paths and then save the image as variables.
        """

        # Creating dataset
        train_set = self.dataset[self.dataset.kind == "train"][
            ["full_path_transformed", "flag"]
        ]
        test_set = self.dataset[self.dataset.kind == "test"][
            ["full_path_transformed", "flag"]
        ]
        val_set = self.dataset[self.dataset.kind == "val"][
            ["full_path_transformed", "flag"]
        ]

        # Creating X and y variables
        self.X_train, self.y_train = self.load_images(train_set)
        self.X_test, self.y_test = self.load_images(test_set)
        self.X_val, self.y_val = self.load_images(val_set)

        # Infortmations
        logging.info(
            "Train data shape: {}, Labels shape: {}".format(
                self.X_train.shape, self.y_train.shape
            )
        )
        logging.info(
            "Test data shape: {}, Labels shape: {}".format(
                self.X_test.shape, self.y_test.shape
            )
        )
        logging.info(
            "Validation data shape: {}, Labels shape: {}".format(
                self.X_val.shape, self.y_val.shape
            )
        )

    def process_data(self, img_path):
        """
        This function wil get the image path, load it, process it and then return it as object.

        :img_path string: The image path.
        :IMAGE_RESOLUTION tuple: Tuple to define the image resolution (x, y, z).
        """

        try:
            # Read the image
            img = cv2.imread(os.path.join("temp", img_path))

            # Transform the color to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Identify lungs
            img = lf.get_lungs(img, padding=20)

            # Scale the image between zero and one
            img = img / 255.0

            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Reshape the image
            img = np.reshape(img, self.IMAGE_RESOLUTION)

            return img
        except:
            # Read the image again
            img = cv2.imread(os.path.join("temp", img_path))
            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Transform the color to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Scale the image between zero and one
            img = img / 255.0

            # Reshape the image
            img = np.reshape(img, IMAGE_RESOLUTION)

            return img

    def load_images(self, df):
        """
        This function will receive a DataFram which contains the information about the images that will be used as dataset.

        :df DataFrame: This DataFrame contains information about the images that will be used as dataset, including the image path.
        """

        # A list to save the image pixels
        data = []
        # A list to save the image flags
        labels = []

        # A for to go through the DataFrame shuffle
        for full_path, flag in df.sample(frac=1).values:
            # Process the image with a auxiliary function and then save in the 'data'
            data.append(self.load_image(full_path))
            # Save the flag in the 'labels'
            labels.append(flag)
        # Return both, data and labels as numpy array
        return np.array(data), np.array(labels)

    def load_image(self, img_path):
        """
        This function will receive an image path and return as numpy array.

        :img_path string: image path.
        """
        # Reading the image already in transformed and using grayscaler
        img = cv2.imread(
            os.path.join(self.transformed_path, img_path), cv2.IMREAD_GRAYSCALE
        )
        # Normalization
        img = img / 255.0
        # Return reshape as ( X, X, 1)
        return np.reshape(img, self.IMAGE_RESOLUTION)

    def download_dataset_from_firebase(self, image_path, storage):
        print(image_path)
        try:
            storage.child(image_path).download(
                filename=os.path.join(self.path_base, image_path), path=os.path.join(self.path_base, image_path), token=token)
            return True
        except:
            return False

    def download_dataset(self):
        storage = self.firebase.storage()
        self.creating_folders()
        self.download_dataset_from_firebase('data_image.csv', storage)
        self.dataset = pd.read_csv(os.path.join("data", "data_image.csv"))
        for img_path_transformed in self.dataset.full_path_transformed:
            # To linux
            aux = img_path_transformed.replace("\\", "/")
            self.download_dataset_from_firebase(aux, storage)

    def creating_folders(self):
        transformed = "transformed"

        path_train_normal = os.path.join("train", "NORMAL")
        path_train_pneu = os.path.join("train", "PNEUMONIA")

        path_test_normal = os.path.join("test", "NORMAL")
        path_test_pneu = os.path.join("test", "PNEUMONIA")

        path_val_normal = os.path.join("val", "NORMAL")
        path_val_pneu = os.path.join("val", "PNEUMONIA")

        Path(os.path.join(self.path_base, transformed, path_train_normal)).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(self.path_base, transformed, path_train_pneu)).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(self.path_base, transformed, path_test_normal)).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(self.path_base, transformed, path_test_pneu)).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(self.path_base, transformed, path_val_normal)).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(self.path_base, transformed, path_val_pneu)).mkdir(
            parents=True, exist_ok=True)


if __name__ == "__main__":
    dataprocessing = DataProcessing()
