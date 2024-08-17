from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Flatten, Input
from keras.applications.vgg16 import VGG16
from src.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config:PrepareBaseModelConfig):
        self.config=config

    @staticmethod
    def save_model(path:Path, model:tf.keras.Model):
        model.save(path)
    
    def get_base_model(self):
        conv_base=VGG16(weights='imagenet', include_top=False, input_shape=self.config.params_image_size)
        input_layer=Input(shape=self.config.params_image_size)
        x=conv_base(input_layer)
        self.model=Model(inputs=input_layer, outputs=x)
        self.model.trainable=False
        self.save_model(path=self.config.base_model_path, model=self.model)
        return self.model
    
    @staticmethod
    def prepare_full_model(model, classes):
        x=Flatten()(model.output)
        x=Dense(units=classes, activation='softmax')(x)
        full_model=Model(inputs=model.input, outputs=x)
        full_model.compile(loss='categorical_crossentropy', optimizer='adam')
        return full_model
    
    def update_base_model(self):
        self.full_model=self.prepare_full_model(
            model=self.model,
            classes=self.config.params_classes
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
        return self.full_model