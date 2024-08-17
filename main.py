from src.pipeline.stage1_prepare_base_model import BasicModelCreatePipeline
from src.pipeline.stage2_train_model import TrainModelPipeline
from src.pipeline.stage3_predict import PredictPipeline
from src import logger


model_creator_pipeline=BasicModelCreatePipeline()
model_creator_pipeline.main()
logger.info('model has been created')


model_trainer_pipeline=TrainModelPipeline()
model_trainer_pipeline.main()
logger.info('model has been trained')

predict=PredictPipeline()
predict.main()