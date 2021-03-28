from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dropout,
)


def cnn10_ann2_pow2_adam(IMG_FORMAT):
    model = Sequential()

    model.add(Conv2D(filters=8, kernel_size=(7, 7), padding='same',
                     activation='relu', input_shape=IMG_FORMAT))
    model.add(Conv2D(filters=8, kernel_size=(7, 7),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(Conv2D(filters=128, kernel_size=(
        3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=1e-5)
    model.compile(loss="binary_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    return model
