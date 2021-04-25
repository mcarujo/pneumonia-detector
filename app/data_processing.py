import logging
import os

import cv2
import numpy as np
import pandas as pd


class DataProcessing:
    def __init__(self):
        """
        Constructor setting some local variables.
        """
        self.IMAGE_RESOLUTION = (250, 250, 1)
        self.BORDER = 30
        self.transformed_path = os.path.join("..", "data")
        DATASET_DIR = os.path.join("data", "data_image.csv")
        self.dataset = pd.read_csv(DATASET_DIR)

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

    def load_images(self, df):
        """
        This function will receive a DataFram which contains the information about the images that will be used as dataset.

        :df DataFrame: This DataFrame contains information about the images that will be used as dataset, including the image path.
        :IMAGE_RESOLUTION tuple: Tuple to define the image resolution (x, y, z).
        :BORDER int: The pixel that will be used to cut the image.
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
        # Reading the image already in transformed and using grayscaler
        img = cv2.imread(
            os.path.join(self.transformed_path, img_path), cv2.IMREAD_GRAYSCALE
        )
        # Normalization
        img = img / 255.0
        # Return reshape as ( X, X, 1)
        return np.reshape(img, self.IMAGE_RESOLUTION)


if __name__ == "__main__":
    """
    DataProcessing test.
    """
    data = DataProcessing()
    data.load_datasets()
