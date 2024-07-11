import tensorflow as tf
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig 
from cnnClassifier.utils.common import save_json # Help me to read the yaml files

import dagshub
dagshub.init(repo_owner='utkarsh-iitbhu', repo_name='mlflow-dvc-classification-tf', mlflow=True)

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
    def _valid_generator(self):
        # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
        # This link has code for ImageDataGenerator 
        
        # List all the params to tune in model
        # These params of dict will go to ImageDataGenerator
        datagenerator_kwargs = dict( # We can add other params as well
            rescale=1./255, # Rescaling the images
            validation_split=0.3 # As we are not having diff test data
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
    
    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        return tf.keras.models.load_model(path)
    
    def evaluation(self): # Evaluate on my data
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator() # Split the data
        # Evaluates the loaded model using the validation data
        self.score = self.model.evaluate(self.valid_generator) # keras.evaluate
        # Stores the evaluation results in self.score
        self.save_score()

    def save_score(self): # 2 metrics are there in score
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
    
    def log_into_mlflow(self):
        mlflow.set_registry_uri(self.config.mlflow_uri) # Track uri
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Retrieves the current tracking URI and parses it to determine the scheme (e.g., http, https, file).
        
        with mlflow.start_run(): # Start run
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            # Model registry does not work with file store
            if tracking_url_type_store != "file":

                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG16Model")
            else:
                mlflow.keras.log_model(self.model, "model")
    