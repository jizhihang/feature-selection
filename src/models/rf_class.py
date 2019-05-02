from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from models.model import Model

class RfClass(Model):
    def __init__(self, model_params):
        super().__init__(model_params,'RF Classifier')

    def build_model(self):
        max_depth = self.model_params['max_depth']
        n_estimators = self.model_params['n_estimators']

        rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

        return rf

        
