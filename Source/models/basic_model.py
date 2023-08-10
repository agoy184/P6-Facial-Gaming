from models.model import Model
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

class BasicModel(Model):
    def _define_model(self, input_shape, categories_count):
        # Your code goes here

        self.model = models.Sequential()
        #keras rescaling layer <- use this
        self.model.add(tf.keras.layers.Rescaling(scale=1./255, offset=0.0))
        #could also change training input
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(categories_count, activation='softmax'))
    
    def _compile_model(self):
        # Your code goes here

        self.model.compile(
            optimizer='rmsprop',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    
