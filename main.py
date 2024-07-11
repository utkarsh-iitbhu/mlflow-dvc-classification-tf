# from src.cnnClassifier import logger
from cnnClassifier import logger # After setting up a local package
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

logger.info("Welcome to our classifier")

try:
    logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e 

STAGE_NAME = "Prepare base model"
try:
    logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=================x")
except Exception as e:
    logger.exception(e)
    raise e 