import logging
import os

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from data_processing import DataProcessing


class ModelTrain:
    def __init__(self):
        """
        Constructor setting DataProcessing.
        """
        self.data = DataProcessing()

    def run(self):
        """
        Start the traning process and then return the results.
        """

        logging.info(f"Runing the train process.")
        # Load the dataset
        logging.info(f"Loading the dataset.")
        self.data.load_datasets()

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
        datagen.fit(self.data.X_train)

        # Models
        logging.info(f"Loading the model.")
        model = self.build_model(self.data.IMAGE_RESOLUTION)

        # EarlyStopping to stop our trainig process when is not nescessary keep training
        callback = EarlyStopping(monitor="loss", patience=3)

        # Define class_weight
        class_weight = {0: 1.95, 1: 0.67}

        # Model fit return the historical metrics of it
        logging.info(f"Starting the fit.")
        history = model.fit(
            datagen.flow(self.data.X_train, self.data.y_train, batch_size=5),
            validation_data=(self.data.X_test, self.data.y_test),
            epochs=50,
            verbose=1,
            callbacks=[callback],
            class_weight=class_weight,
        )
        print(history.history)

        logging.info(f"Generating metrics.")
        # Predicting the classes model
        y_pred = model.predict(self.data.X_test, batch_size=4)

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
            self.data.y_test,
            y_pred.reshape(1, -1)[0],
            y_pred_class.reshape(1, -1)[0].astype(int),
        )

    def build_model(self, IMAGE_RESOLUTION):
        """
        Function to return the convolutional neural network.
        """

        model = Sequential()

        model.add(
            Conv2D(
                filters=8,
                kernel_size=(7, 7),
                padding="same",
                activation="relu",
                input_shape=IMAGE_RESOLUTION,
            )
        )
        model.add(
            Conv2D(filters=8, kernel_size=(7, 7), padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(
            Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=16, kernel_size=(5, 5), padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(3, 3)))

        model.add(
            Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(
            Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")
        )
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation="sigmoid"))

        optimizer = Adam(lr=0.0001, decay=1e-5)
        model.compile(
            loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"]
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
        fig.write_image(os.path.join("static", "train", "train_graph.png"))

    def shap_values(self, model):
        """
        This function will receive the model trained, create the Explainer and save in a path.

        :model: The trained model.
        """

        # Creating the DeepExplainer
        logging.info(f"Creating the DeepExplainer.")
        e = shap.DeepExplainer(model, self.data.X_val)
        joblib.dump(self.data.X_val, "model/shap_dataset.joblib")

        # Generating the shap values
        logging.info(f"Creating the shap values.")
        shap_values = e.shap_values(self.data.X_val)

        # Plot the image explaining the predictions
        logging.info(f"Creating the shap image.")
        fig = shap.image_plot(shap_values, self.data.X_val, show=False)

        # Saving the image
        logging.info(f"Saving shap graph at 'static/train/shap_graph.png'.")
        plt.savefig(os.path.join("static", "train", "shap_graph.png"))


class ModelPredict:
    def __init__(self):
        """
        Constructor loading model and DataProcessing.
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
        img = self.data.process_data(image_path)
        if img is False:
            return False
        img = np.array(img).reshape(1, *self.data.IMAGE_RESOLUTION)

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

    def define_info_label(self, prediction):
        """
        Function will get the prediction and tranform in labels.

        :prediction float: The prediction probability to be Pneumonia.
        """
        if prediction < 0.3:
            return "Strong Normal"
        elif prediction < 0.5:
            return "Weak Normal"
        elif prediction < 0.7:
            return "Weak Pneumonia"
        else:
            return "Strong Pneumonia"

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
        dataset_shap = joblib.load(os.path.join("model", "shap_dataset.joblib"))

        # DeepExplainer tensorflow
        explainer = shap.DeepExplainer(self.model, dataset_shap)

        # Shap Values
        shap_values = explainer.shap_values(img)

        # Plot the image explaining the predictions
        fig = shap.image_plot(shap_values, img, labels=labels, show=False)

        # Saving the image
        plt.savefig(os.path.join("static", "predict", "shap_" + image_path))
