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
    
