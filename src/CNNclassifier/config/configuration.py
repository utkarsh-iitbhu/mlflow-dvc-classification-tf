from cnnClassifier.constants import * # This will import the paths for config and param yaml
from cnnClassifier.utils.common import read_yaml, create_directories # Help me to read the yaml files
from cnnClassifier.entity.config_entity import (DataIngestionConfig)

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
