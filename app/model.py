import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from plotly import graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_processing import DataProcessing


class ModelTrain:
    def __init__(self):
        """
        Constructor setting some local variables and dataset loading.
        """

        DATASET_DIR = "../data/data_image.csv"
        self.dataset = pd.read_csv(DATASET_DIR)
        self.data = DataProcessing()

    def load_datasets(self):
        """
        Load the dataset with the images paths and then save the image as variables.
        """

        # Creating dataset
        train_set = self.dataset[self.dataset.kind ==
                                 "train"][["full_path", "flag"]]
        test_set = self.dataset[self.dataset.kind ==
                                "test"][["full_path", "flag"]]
        val_set = self.dataset[self.dataset.kind ==
                               "val"][["full_path", "flag"]]

        # Creating X and y variables
        self.X_train, self.y_train = self.data.compose_dataset(train_set)
        self.X_test, self.y_test = self.data.compose_dataset(test_set)
        self.X_val, self.y_val = self.data.compose_dataset(val_set)

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

    def run(self):
        """
        Start the traning process and then return the results.
        """

        logging.info(f"Runing the train process.")
        # Load the dataset
        logging.info(f"Loading the dataset.")
        self.load_datasets()

        # Define ImageDataGenerator
        datagen = ImageDataGenerator(
            rotation_range=20,
            zoom_range=0.3,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=False,
            vertical_flip=False,
        )

        # Fit generator on our train features
        datagen.fit(self.X_train)

        # Models
        logging.info(f"Loading the model.")
        model = self.build_model(self.data.IMG_FORMAT)

        # EarlyStopping to stop our trainig process when is not nescessary keep training
        callback = EarlyStopping(monitor="loss", patience=3)

        # Define class_weight
        class_weight = {0: 1.95, 1: 0.67}

        # Model fit return the historical metrics of it
        logging.info(f"Starting the fit.")
        history = model.fit(
            datagen.flow(self.X_train, self.y_train, batch_size=5),
            validation_data=(self.X_test, self.y_test),
            epochs=2,
            verbose=1,
            callbacks=[callback],
            class_weight=class_weight,
        )

        logging.info(f"Generating metrics.")
        # Predicting the classes model
        y_pred = model.predict(self.X_test, batch_size=4)

        # Predicting the classes model
        y_pred_class = y_pred.round()

        # Saving the model
        logging.info(f"Saving the model.")
        model.save("model")

        # Saving training graph
        logging.info(f"Save the plot training.")
        self.save_plot_training(history)

        # Saving shap values for validation dataset
        logging.info(f"Using Shap to explain the predictions.")
        self.shap_values(model)

        # Returning traning metrics
        logging.info(f"Returning metrics.")
        return self.generate_metrics(
            self.y_test,
            y_pred.reshape(1, -1)[0],
            y_pred_class.reshape(1, -1)[0].astype(int),
        )

    def build_model(self, IMG_FORMAT):
        """
        Function to return the convolutional neural network.
        """

        # Empty convolutional neural network
        model = Sequential()

        # Convolutional layers with the input layer
        model.add(
            Conv2D(
                filters=10,
                kernel_size=(7, 7),
                padding="same",
                activation="relu",
                input_shape=IMG_FORMAT,
            )
        )
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(
            Conv2D(filters=20, kernel_size=(5, 5),
                   padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(
            Conv2D(filters=30, kernel_size=(3, 3),
                   padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=40, kernel_size=(3, 3),
                   padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=50, kernel_size=(3, 3),
                   padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(
            Conv2D(filters=60, kernel_size=(3, 3),
                   padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())

        # Dense layers
        model.add(Dense(200, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(100, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(50, activation="relu"))
        model.add(Dropout(0.2))

        # Output layer
        model.add(Dense(1, activation="sigmoid"))

        # Model compile and optimizer
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        return model

    def generate_metrics(self, y_true, y_pred_class, y_pred):
        """
        This function will receive the real labels and the predictions to generate metrics then return the image and save in a path.

        :y_true list: A list with the real labels [int].
        :y_pred_class list: A list with the classes predicted [int].
        :y_pred list: A list with the probability of both classes [[float,float]].
        """

        # Generating metrics with scikit-learn
        ac = accuracy_score(y_true, y_pred)
        ll = log_loss(y_true, y_pred_class)
        f1 = f1_score(y_true, y_pred, zero_division=1)
        ps = precision_score(y_true, y_pred)
        rc = recall_score(y_true, y_pred)

        logging.info(f"Training results")
        logging.info(f"Accuracy -> {ac}.")
        logging.info(f"Log loss -> {ll}.")
        logging.info(f"F1 score -> {f1}.")
        logging.info(f"Precision -> {ps}.")
        logging.info(f"Recall -> {rc}.")

        return {"ac": ac, "ll": ll, "f1": f1, "ps": ps, "rc": rc}

    def save_plot_training(self, history):
        """
        This function will receive the training information then save in a path.

        :history TensorFlowHistory: A object which contains the training information.
        """

        # Creating the plotly figure object
        fig = go.Figure()
        # Adding accuracy line
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(history.history["accuracy"])),
                y=history.history["accuracy"],
                text=np.arange(len(history.history["accuracy"])),
                mode="lines",
                name="accuracy",
            )
        )
        # Adding val_accuracy line
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(history.history["val_accuracy"])),
                text=np.arange(len(history.history["val_accuracy"])),
                y=history.history["val_accuracy"],
                mode="lines",
                name="val_accuracy",
            )
        )
        # Formating the graph
        fig.update_layout(
            title="Training", xaxis_title="Epochs", yaxis_title="Accuracy"
        )
        # Saving the image in png
        logging.info(f"Saving train graph at 'static/train/train_graph.png'.")
        fig.write_image("static/train/train_graph.png")

    def shap_values(self, model):
        """
        This function will receive the model trained, create the Explainer and save in a path.

        :model: The trained model.
        """

        logging.info(f"Creating the DeepExplainer.")
        e = shap.DeepExplainer(model, self.X_val)

        logging.info(f"Creating the shap values.")
        shap_values = e.shap_values(self.X_val)

        # Plot the image explaining the predictions
        logging.info(f"Creating the shap image.")
        fig = shap.image_plot(shap_values, self.X_val,
                              show=False, matplotlib=True)

        # Saving the image
        logging.info(f"Saving shap graph at 'static/train/shap_graph.png'.")
        plt.savefig("/static/train/shap_graph.png")


class ModelPredict:
    def __init__(self):
        """
        Constructor loading model and dataset loading.
        """

        logging.info(f"Initializing ModelPredict.")
        self.model = keras.models.load_model("model")
        self.data = DataProcessing()

    def predict(self, image_path):
        """
        Function to prediction the input and return the prediction and the explanation.

        :image_path str: Image path to be predicted.
        """

        logging.info(f"Predicting an image: {image_path}.")

        # Image processing
        img = self.data.process_data(image_path, False)

        if img is False:
            return False

        img = np.array(img).reshape(1, *self.data.IMG_FORMAT)

        # Predicting the image
        prediction = self.model.predict(img)[0][0]
        logging.info(f"The image: {image_path} was predict as: {prediction}.")

        # Explaining the image
        logging.info(
            f"Explaining the prediction of the image: {image_path} that was predict as: {prediction}."
        )
        self.shap_values(img, image_path, prediction)

        return prediction

    def define_label(self, prediction):
        """
        Function will get the prediction and tranform it as label.

        :prediction float: The prediction probability to be Pneumonia.
        """

        return "Predict Pneumonia" if prediction > 0.5 else "Predict Normal"

    def shap_values(self, img, image_path, prediction):
        """
        Function to explain the prediction and save the shap file in the path.

        :img NumpyArray: Image already load as matrix.
        :image_path str: Image path to be predicted.
        :prediction float: The prediction in probability.
        """

        # Creating a string list with the labels
        labels = np.array(self.define_label(prediction)).reshape(-1, 1)

        # Loading the validation dataset
        dataset_shap = joblib.load("model/shap_dataset.joblib")

        # DeepExplainer tensorflow
        explainer = shap.DeepExplainer(self.model, dataset_shap)

        # Shap Values
        shap_values = explainer.shap_values(img)

        # Plot the image explaining the predictions
        fig = shap.image_plot(shap_values, img, labels=labels, show=False)

        # Saving the image
        plt.savefig("static/predict/shap_" + image_path)


if __name__ == "__main__":
    """
    Model training test.
    """
    test = ModelTrain()
    print(test.run())
    test = ModelPredict()
    print(test.predict("person1946_bacteria_4874.jpeg"))
