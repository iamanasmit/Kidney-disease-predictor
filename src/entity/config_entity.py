#general structure of the project

from dataclasses import dataclass
from pathlib import Path

from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_include_top: bool
    params_classes: int
    param_weights: str

@dataclass(frozen=True)
class PrepareTrainingConfig:
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_is_augmented: bool

@dataclass(frozen=True)
class PreparePredictionConfig:
    root_dir: Path
    trained_model_path: Path
    prediction_data: Path