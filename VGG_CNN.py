from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D


class VGG_CNN:
    pass

    def __init__(self):
        pass

    def Modal(self):
        input_shape = (224, 224, 1)

        model = Sequential()

        model.add(Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', ))
        model.add(Conv2D(512, (3, 3), activation='relu', padding='same', ))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4096, activation='relu'))
        model.add(Dense(4, activation='softmax'))


        model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

        return model