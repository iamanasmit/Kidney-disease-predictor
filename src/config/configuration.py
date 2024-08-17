#config of the model

from src.constants import CONFIG_FILE_PATH, PARAM_FILE_PATH, TRAINING_DATA_PATH, PREDICTION_DATA_PATH
from src.utils.common import read_yaml, create_directories
from src.entity.config_entity import PrepareBaseModelConfig, PrepareTrainingConfig, PreparePredictionConfig
from pathlib import Path

class ConfigManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath= PARAM_FILE_PATH):
        
        self.config=read_yaml(config_filepath)
        self.params=read_yaml(params_filepath)
        
        create_directories(self.config.artifacts_root)
        create_directories(self.config.training.root_dir)

    def get_prepare_base_model_config(self) ->PrepareBaseModelConfig:
        config=self.config.prepare_base_model

        model_config=PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_include_top=self.params.INCLUDE_TOP,
            params_classes=self.params.CLASSES,
            param_weights=self.params.WEIGHTS
        )

        return model_config
    
    def get_training_configs(self) ->PrepareTrainingConfig:
        training=self.config.training
        prepare_base_model=self.config.prepare_base_model

        training_config=PrepareTrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=TRAINING_DATA_PATH,
            params_epochs=self.params.EPOCHS,
            params_is_augmented=self.params.AUGMENTATION
        )
        return training_config
    
    def get_prediction_configs(self) -> PreparePredictionConfig:
        prediction_config=PreparePredictionConfig(
            root_dir=self.config.prepare_base_model.root_dir,
            trained_model_path=self.config.training.trained_model_path,
            prediction_data= PREDICTION_DATA_PATH
        )

        return prediction_config
    

