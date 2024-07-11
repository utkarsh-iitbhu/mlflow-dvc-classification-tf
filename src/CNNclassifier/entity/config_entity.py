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