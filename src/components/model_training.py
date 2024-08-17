#add to components
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.entity.config_entity import PrepareTrainingConfig
import os


class Trainer:
    def __init__(self, config: PrepareTrainingConfig):
        self.config=config

    def get_base_model(self):
        self.model=tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
    
    def load_data(self):
        self.normal_path=os.path.join(self.config.training_data, 'Normal')
        self.tumor_path=os.path.join(self.config.training_data, 'Tumor')
        self.stone_path=os.path.join(self.config.training_data, 'Stone')
        self.cyst_path=os.path.join(self.config.training_data, 'Cyst')

        self.normal_labels=[]#encode to (1, 0, 0, 0)
        self.tumor_labels=[]#encode to (0, 1, 0, 0)
        self.stone_labels=[]#encode to (0, 0, 1, 0)
        self.cyst_labels=[]#encode to (0, 0, 0, 1)

        self.normal_tensors=[]
        self.tumor_tensors=[]
        self.stone_tensors=[]
        self.cyst_tensors=[]

        for filename in os.listdir(self.normal_path):
            img=cv2.imread(os.path.join(self.normal_path, filename),1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img, (224, 224))
            self.normal_tensors.append(img)
            self.normal_labels.append([1, 0, 0, 0])

        for filename in os.listdir(self.tumor_path):
            img=cv2.imread(os.path.join(self.tumor_path, filename),1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img, (224, 224))
            self.tumor_tensors.append(img)
            self.tumor_labels.append([0, 1, 0, 0])

        for filename in os.listdir(self.stone_path):
            img=cv2.imread(os.path.join(self.stone_path, filename),1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img, (224, 224))
            self.stone_tensors.append(img)
            self.stone_labels.append([0, 0, 1, 0])

        for filename in os.listdir(self.cyst_path):
            img=cv2.imread(os.path.join(self.cyst_path, filename),1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img=cv2.resize(img, (224, 224))
            self.cyst_tensors.append(img)
            self.cyst_labels.append([0, 0, 0, 1])
        
        rng = np.random.default_rng(42)

        self.normal_tensors=np.array(self.normal_tensors, dtype=np.uint8)
        self.tumor_tensors=np.array(self.tumor_tensors, dtype=np.uint8)
        self.stone_tensors=np.array(self.stone_tensors, dtype=np.uint8)
        self.cyst_tensors=np.array(self.cyst_tensors, dtype=np.uint8)

        self.x=np.concatenate((self.normal_tensors, self.tumor_tensors, self.stone_tensors, self.cyst_tensors), axis=0)
        rng.shuffle(self.x)

        del self.normal_tensors
        del self.tumor_tensors
        del self.stone_tensors
        del self.cyst_tensors

        self.normal_labels=np.array(self.normal_labels, dtype=np.uint8)
        self.tumor_labels=np.array(self.tumor_labels, dtype=np.uint8)
        self.stone_labels=np.array(self.stone_labels, dtype=np.uint8)
        self.cyst_labels=np.array(self.cyst_labels, dtype=np.uint8)

        self.y=np.concatenate((self.normal_labels, self.tumor_labels, self.stone_labels, self.cyst_labels), dtype=np.int8)
        rng.shuffle(self.y)

        del self.normal_labels
        del self.tumor_labels
        del self.stone_labels
        del self.cyst_labels

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def train(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.summary()
        self.model.fit(self.x, self.y, epochs=self.config.params_epochs)
        self.save_model(self.config.trained_model_path, self.model)
        