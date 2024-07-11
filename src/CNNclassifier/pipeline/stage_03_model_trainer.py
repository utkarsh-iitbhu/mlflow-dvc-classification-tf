from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.model_trainer import Training
from cnnClassifier import logger 

STAGE_NAME = "Model training"

class ModelTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model() # Call base model path
        training.train_valid_generator() # Get data
        training.train() # Train it and save it

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> Stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx=================x")
    except Exception as e:
        logger.exception(e)
        raise e 