from src.components.model_training import Trainer
from src.config.configuration import ConfigManager

class TrainModelPipeline:
    def __init__(self):
        pass

    def main(self):
        training_config_generator=ConfigManager()
        training_config=training_config_generator.get_training_configs()
        trainer=Trainer(training_config)
        trainer.get_base_model()
        trainer.load_data()
        trainer.train()