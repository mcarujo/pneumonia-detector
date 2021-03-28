from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
)


def cnn5_ann5_fibonacci_adam(IMG_FORMAT):
    model = Sequential()
    model.add(
        Conv2D(
            filters=55,
            kernel_size=(7, 7),
            padding="same",
            activation="relu",
            input_shape=IMG_FORMAT,
        )
    )
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Conv2D(filters=34, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=21, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=13, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=8, kernel_size=(3, 3),
                     padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(144, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Dense(89, activation="relu"))
    model.add(Dense(55, activation="relu"))
    model.add(Dense(34, activation="relu"))
    model.add(Dense(21, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy",
                  optimizer="adam", metrics=["accuracy"])
    return model
