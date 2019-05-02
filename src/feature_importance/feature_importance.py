import numpy as np
import pandas as pd

class FeatureImportance:
    def __init__(self, model_type):
        self.model_type = model_type
    
    def calc_feature_importance(self, model, x_train, x_test, y_test, feature_names):
        pass