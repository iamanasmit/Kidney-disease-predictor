from src.components.predict import *
from src.config.configuration import ConfigManager

class PredictPipeline:
    def __init__(self):
        pass

    def main(self):
        cmanager=ConfigManager()
        prediction_config=cmanager.get_prediction_configs()
        p=Predictor(prediction_config)
        p.load_model()
        p.load_data()
        y=p.estimate()
        print(y.argmax())

if __name__=='main':
    obj=PredictPipeline()
    obj.main()