import os
from cnnClassifier.constants import * # This will import the paths for config and param yaml
from cnnClassifier.utils.common import read_yaml, create_directories # Help me to read the yaml files
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig)
# There is no functional difference between using parentheses or not. These two import statements are functionally equivalent

class ConfigurationManager: # Read the config and param file paths
    def __init__(self, config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        
        # This will create the directory for dataset
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self)-> DataIngestionConfig: # Returning 
        config = self.config.data_ingestion # This key has all the values
        create_directories([config.root_dir]) # This will create the root directory
        
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir, # Storing all the variables
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        ) 
        return data_ingestion_config

    # Configuration.py
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model # Called from config.yaml
        create_directories([config.root_dir]) # Dir for base model
        
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir = Path(config.root_dir),
            base_model_path = Path(config.base_model_path),
            updated_base_model_path = Path(config.updated_base_model_path),
            # Update the params now 
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES
        )

        return prepare_base_model_config

    # Training config from 03_model_trainer
    def get_training_config(self)->TrainingConfig:
        training = self.config.training # Taken from config.yaml
        prepare_base_model = self.config.prepare_base_model # Taken from config
        params = self.params # Taken from parans.yaml 
        training_data = os.path.join(self.config.data_ingestion.unzip_dir,"Chest-CT-Scan-data")
        create_directories([Path(training.root_dir)])
        
        training_config = TrainingConfig( # Just adding correct path for all
            root_dir = Path(training.root_dir),
            trained_model_path= Path(training.trained_model_path),
            updated_base_model_path= Path(prepare_base_model.updated_base_model_path),
            training_data= Path(training_data),
            params_epochs= params.EPOCHS,
            params_batch_size= params.BATCH_SIZE,
            params_is_augmentation= params.AUGMENTATION,
            params_image_size= params.IMAGE_SIZE 
        )
        
        return training_config