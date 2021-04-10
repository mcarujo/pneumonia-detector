import cv2
import numpy as np


class DataProcessing:
    def __init__(self):
        """
        Constructor setting some local variables.
        """
        self.IMAGE_RESOLUTION = (224, 224, 1)
        self.BORDER = 30

    def process_data(self, img_path, to_train=True):
        """
        This function wil get the image path, load it and then return it as object.

        :img_path string: The image path.
        :to_train bool: Define the path for train and prediction.
        """

        # Read the image
        if to_train:
            img_path = "../" + img_path
        else:
            img_path = "temp/" + img_path

        try:
            # Read the image
            img = cv2.imread(img_path)
            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Transform the color to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Identify lungs
            img = lf.get_lungs(img, padding=0)

            # Scale the image between zero and one
            img = img / 255.0

            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Reshape the image
            img = np.reshape(img, self.IMAGE_RESOLUTION)

            return img
        except:
            # Read the image again
            img = cv2.imread(img_path)
            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Transform the color to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Scale the image between zero and one
            img = img / 255.0

            # Cut the image
            img = img[
                self.BORDER: self.IMAGE_RESOLUTION[0] - self.BORDER, self.BORDER: self.IMAGE_RESOLUTION[0] - self.BORDER
            ]

            # Resize the image
            img = cv2.resize(img, self.IMAGE_RESOLUTION[:2])

            # Reshape the image
            img = np.reshape(img, self.IMAGE_RESOLUTION)

            return img

    def compose_dataset(self, df):
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
            data.append(self.process_data(full_path))
            # Save the flag in the 'labels'
            labels.append(flag)
        # Return both, data and labels as numpy array
        return np.array(data), np.array(labels)
