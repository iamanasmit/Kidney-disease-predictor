from src.components.prepare_base_model import PrepareBaseModel
from src.config.configuration import ConfigManager
from src import logger

STAGE_NAME='preparing base model'

class BasicModelCreatePipeline:
    def __init__(self):
        pass
    def main(self):
        _config=ConfigManager()
        model_config=_config.get_prepare_base_model_config()
        model_generator_obj=PrepareBaseModel(model_config)
        conv_networks=model_generator_obj.get_base_model()
        full_model=model_generator_obj.update_base_model()

if __name__=='__main__':
    try:
        logger.info(f'stage {STAGE_NAME} has started')
        obj=BasicModelCreatePipeline()
        obj.main()
        logger.info(f'stage {STAGE_NAME} completed successfully')
    except:
        print('some error has occured')