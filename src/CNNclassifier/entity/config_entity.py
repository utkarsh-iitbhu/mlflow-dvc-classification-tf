from dataclasses import dataclass
from pathlib import Path

# These are the configurations it will return 
# frozen=True will not allow any other functionalities to be added 
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_URL: str 
    local_data_file: Path
    unzip_dir: Path
# This will go in my entity directory of code

# This config is for Preparing our model
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path # From config.yaml
    updated_base_model_path: Path # From config.yaml
    params_image_size: list # Add params from params.yaml
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    
# This is our Model training class 
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path 
    trained_model_path: Path # Taken from the config.yaml
    updated_base_model_path: Path # from artifacts/prepare_base_model/updated_base_model
    training_data: Path # from artifacts/data_ingestion 
    params_epochs: int # Rest are taken from the params.yaml
    params_batch_size: int 
    params_is_augmentation: bool 
    params_image_size: list 
    
# This is to evaluate model
@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path # Model path
    training_data: Path # Data Path
    all_params: dict # Get all the params as a dict
    mlflow_uri: str # mlflow uri to target
    params_image_size: list 
    params_batch_size: int