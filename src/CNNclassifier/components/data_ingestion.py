import os 
import zipfile 
import gdown 
from cnnClassifier import logger 
from cnnClassifier.utils.common import get_size 
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config # This will take the config return class
    
    def download_file(self)->str:
        """ Fetch data from URL """ 
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion",exist_ok=True)
            logger.info(f"Downloading the data from {dataset_url} into file {zip_download_dir}")
            # This will find link, make a dir at makedirs and then download the zip file
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.info(f"Downloaded the data from {dataset_url} into file {zip_download_dir}")
        
        except Exception as e:
            raise e
        
    def extract_zip_files(self):
        """ Extract the zip file into a folder """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path,exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file,'r') as zip_ref:
            zip_ref.extractall(unzip_path)