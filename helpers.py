"""
This file just to save some functions to help keep the notebooks clean.  
"""
import os
import cv2
import numpy as np
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)


def plot_training(history, path):
    """
    This function will receive the training information then return the image and save in a path.

    :history TensorFlowHistory: A object which contains the training information.
    :path string: The path to save the graph.
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
    fig.update_layout(title="CNN training curve plot.", xaxis_title="Epochs", yaxis_title="Score", title_font_size=24)
    # Saving the image in png
    fig.write_image(os.path.join(path, "train_graph.png"))

    # Showing the image
    fig.show()


def metrics(y_true, y_pred_class, y_pred, path):
    """
    This function will receive the real labels and the predictions to generate metrics then return the image and save in a path.

    :y_true list: A list with the real labels [int].
    :y_pred_class list: A list with the classes predicted [int].
    :y_pred list: A list with the probability of both classes [[float,float]].
    :path string: The path to save the graph.
    """
    # Generating metrics with scikit-learn
    ac = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    ps = precision_score(y_true, y_pred)
    mc = confusion_matrix(y_true, y_pred)
    rc = recall_score(y_true, y_pred)

    # Setting the labels for confusion matrix and metrics tables
    header = ["Metric", "Accuracy", "Loss(log)", "F1", "Precision", "Recall"]
    score = [
        "Score",
        round(ac, 3),
        round(ll, 3),
        round(f1, 3),
        round(ps, 3),
        round(rc, 3),
    ]
    x = ["Real Normal", "Real Pneumonia"]
    y = ["Predict Normal", "Predict Pneumonia"]

    # Creating the metrics table
    fig = ff.create_table([header, score], height_constant=20)
    # Saving the image in png
    fig.write_image(os.path.join(path, "metrics.png"))

    # Showing the image
    fig.show()

    # Creating the confusion matrix
    fig = ff.create_annotated_heatmap(z=mc, x=x, y=y, colorscale="Blues")
    # Saving the image in png
    fig.write_image(os.path.join(path, "confusion_matrix.png"))

    # Showing the image
    fig.show()


def load_images(df, IMAGE_RESOLUTION, ROOT_PATH='data/'):
    """
    This function will receive a DataFram which contains the information about the images that will be used as dataset.
    ROOT_PATH defined as 'data/' due to the local project.
    :df DataFrame: This DataFrame contains information about the images that will be used as dataset, including the image path.
    """

    # A list to save the image pixels
    data = []
    # A list to save the image flags
    labels = []

    # A for to go through the DataFrame shuffle
    for full_path, flag in df.sample(frac=1).values:
        # Process the image with a auxiliary function and then save in the 'data'
        data.append(load_image(full_path,IMAGE_RESOLUTION, ROOT_PATH))
        # Save the flag in the 'labels'
        labels.append(flag)
    # Return both, data and labels as numpy array
    return np.array(data), np.array(labels)

def load_image(img_path, IMAGE_RESOLUTION, ROOT_PATH):
    """
    This function will receive an image path and return as numpy array.

    :img_path string: image path.
    """
    # Reading the image already in transformed and using grayscaler
    img = cv2.imread(ROOT_PATH + img_path, cv2.IMREAD_GRAYSCALE
    )
    # Normalization
    img = img / 255.0
    # Return reshape as ( X, X, 1)
    return np.reshape(img, IMAGE_RESOLUTION)
