import os 
import time
import tensorflow as tf 
from zipfile import ZipFile 
import urllib.request as request 
from pathlib import Path 
from cnnClassifier.entity.config_entity import TrainingConfig

class Training:
    def __init__(self,config:TrainingConfig):
        self.config = config
        
    def get_base_model(self): # Take the upd model path
        self.model = tf.keras.models.load_model(
            self.config.updated_base_model_path
        )
        
    def train_valid_generator(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # This link has code for ImageDataGenerator 
        
        # List all the params to tune in model
        # These params of dict will go to ImageDataGenerator
        datagenerator_kwargs = dict( # We can add other params as well
            rescale=1./255, # Rescaling the images
            validation_split=0.2 # As we are not having diff test data
        )
        
        # This is for flow_from_directory
        dataflow_kwargs = dict( # This is same for train and validation sets
            target_size= self.config.params_image_size[:-1],#(224,224)
            batch_size= self.config.params_batch_size, # 16
            interpolation="bilinear", # This is for resizing
        )
        
        # Now create your datagenerator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs # Fill in the dict of values here
        )
        
        # Now create the flow_from_directory | how to get data from dir
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory= self.config.training_data, # Comes from Inheritance
            subset="validation", # This is for validation data
            shuffle=False,
            **dataflow_kwargs # This will load the other params
        )
        
        # If augmentation is true apply it
        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                **datagenerator_kwargs
                )
        else: # Else keep the same generator
            train_datagenerator = valid_datagenerator
        
        # Now we took out 20% of data for val, rest goes to training
        self.train_generator = train_datagenerator.flow_from_directory(
            directory= self.config.training_data, # Same dir
            subset="training", # Subset is training
            shuffle=True, # Shuffle my train data
            **dataflow_kwargs # Call rest of the dict
            )
    @staticmethod
    def save_model(path: Path, model:tf.keras.Model):
        model.save(path) # Save our model in given path
        
    # Training begins
    def train(self):
        self.steps_per_epoch = self.train_generator.samples//self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples//self.valid_generator.batch_size
        
        self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            # Pass validation data and its steps per epochs
            validation_data=self.valid_generator,
            validation_steps=self.validation_steps,
        )
        
        # Now trained model we have to save
        self.save_model(
            path= self.config.trained_model_path,
            model= self.model)