"""
This file just to save some functions to help keep the notebooks clean.  
"""
import os
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
    fig.update_layout(title="Training", xaxis_title="Epochs", yaxis_title="Accuracy")
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
