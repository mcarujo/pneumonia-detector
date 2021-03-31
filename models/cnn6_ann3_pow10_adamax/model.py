"""
This file is responsible to define the convolutional neural network that will be used as model, which is returned by a function.  
"""

from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential


def cnn6_ann3_pow10_adamax(IMG_FORMAT):
    model = Sequential()
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
    model.add(Conv2D(filters=20, kernel_size=(5, 5),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=30, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=40, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=50, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=60, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(200, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adamax", metrics=["accuracy"])
    return model
