from sklearn.linear_model import LogisticRegression
import pandas as pd
from models.model import Model

class LogReg(Model):
    def __init__(self, model_params):
        super().__init__(model_params,'Log Reg Classifier')

    def build_model(self):
        max_iter = self.model_params['max_iter']

        lr = LogisticRegression(max_iter=max_iter)

        return lr

        
