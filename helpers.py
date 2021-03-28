# File just to save some functions to help keep the notebooks clean
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly import figure_factory as ff
from plotly import graph_objects as go
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_score, recall_score)


def process_data(img_path, IMAGE_RESOLUTION, BORDER):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMAGE_RESOLUTION[:2])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = np.reshape(img, IMAGE_RESOLUTION)
    img = img[BORDER:IMAGE_RESOLUTION[0]-BORDER,
              BORDER:IMAGE_RESOLUTION[0]-BORDER]
    return img


def compose_dataset(df, IMAGE_RESOLUTION, BORDER):
    data = []
    labels = []

    for full_path, flag in df.sample(frac=1).values:
        data.append(process_data(full_path, IMAGE_RESOLUTION, BORDER))
        labels.append(flag)

    return np.array(data), np.array(labels)


def plot_training(history, path):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(history.history["accuracy"])),
            y=history.history["accuracy"],
            text=np.arange(len(history.history["accuracy"])),
            mode="lines",
            name="accuracy",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(history.history["val_accuracy"])),
            text=np.arange(len(history.history["val_accuracy"])),
            y=history.history["val_accuracy"],
            mode="lines",
            name="val_accuracy",
        )
    )
    fig.update_layout(title="Training", xaxis_title="Epochs",
                      yaxis_title="Accuracy")
    fig.write_image(path + '\\train_graph.png')
    fig.show()


def metrics(y_true, y_pred_class, y_pred, path):
    ac = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred, zero_division=1)
    ps = precision_score(y_true, y_pred)
    mc = confusion_matrix(y_true, y_pred)
    rc = recall_score(y_true, y_pred)

    header = ["Metric", "Accuracy", "Loss(log)", "F1", "Precision", "Recall"]
    score = [
        "Score",
        round(ac, 3),
        round(ll, 3),
        round(f1, 3),
        round(ps, 3),
        round(rc, 3),
    ]

    x = ["Real 0", "Real 1"]
    y = ["Predict 0", "Predict 1"]

    fig = ff.create_table([header, score], height_constant=20)
    fig.write_image(path + '\metrics.png')
    fig.show()

    fig = ff.create_annotated_heatmap(z=mc, x=x, y=y, colorscale="Blues")
    fig.write_image(path+'\confusion_matrix.png')
    fig.show()
