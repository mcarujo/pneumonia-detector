#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
from pandas_profiling import ProfileReport


# # Paths
#
# Define all paths for images in a UNIX evironment...


raw = "raw"
transformed = "transformed"

path_train_normal = os.path.join("train", "NORMAL")
path_train_pneu = os.path.join("train", "PNEUMONIA")

path_test_normal = os.path.join("test", "NORMAL")
path_test_pneu = os.path.join("test", "PNEUMONIA")

path_val_normal = os.path.join("val", "NORMAL")
path_val_pneu = os.path.join("val", "PNEUMONIA")


# # Creating auxiliary function


def creat_dataframe(base_path, kind, flag):
    """
    Will read all information from a folder and return as DataFrame.
    """
    # Getting all images names
    train_normal_imgs = os.listdir(os.path.join(raw, base_path))
    # Creating a DataFrame
    aux = pd.DataFrame(train_normal_imgs)
    aux.columns = ["image_name"]

    # Creating new columns from
    aux["full_path_raw"] = aux.image_name.apply(
        lambda x: os.path.join(raw, base_path, x)
    )
    aux["full_path_transformed"] = aux.image_name.apply(
        lambda x: os.path.join(transformed, base_path, x)
    )
    aux["kind"] = kind
    aux["flag"] = flag
    return aux


# # Grid with all options


grid = [
    (path_train_normal, "train", 0),
    (path_train_pneu, "train", 1),
    (path_test_normal, "test", 0),
    (path_test_pneu, "test", 1),
    (path_val_normal, "val", 0),
    (path_val_pneu, "val", 1),
]


# # Creating DataFrame

dataset_images = pd.concat([creat_dataframe(*el) for el in grid])
dataset_images.to_csv("data_image.csv", index=False)
dataset_images.to_csv(os.path.join("..", "app", "data", "data_image.csv"), index=False)


# # PorfileReport


profile = ProfileReport(dataset_images, title="Pandas Profiling Report")


profile.to_file(os.path.join("..", "app", "templates", "dataset.html"))
