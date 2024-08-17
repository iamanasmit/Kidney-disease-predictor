import cv2
import numpy as np
import os
from src.entity.config_entity import PreparePredictionConfig
import tensorflow as tf


class Predictor:
    def __init__(self, config: PreparePredictionConfig):
        self.config=config
    
    def load_model(self):
        self.model=tf.keras.models.load_model(self.config.trained_model_path)

    def load_data(self):
        self.data=[]
        for filename in os.listdir(self.config.prediction_data):
            filepath=os.path.join(self.config.prediction_data, filename)
            img=cv2.imread(filepath, 1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img, (224, 224))
            self.data.append(img)
        self.data=np.array(self.data, dtype=np.uint8)

    def estimate(self):
        self.y=self.model.predict(self.data)
        return self.y
    