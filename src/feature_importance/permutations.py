import numpy as np
import pandas as pd

from feature_importance.feature_importance import FeatureImportance
from evaluators.evaluator import Evaluator
from models.model import Model

class Permutation(FeatureImportance):
    def __init__(self, model_type):
        super().__init__(model_type)

    def calc_feature_importance(self, model, x_train, x_test, y_test, feature_names):
        test_df = pd.DataFrame(y_test)
        cols = test_df.columns.values.tolist()
        if len(cols)==1:
            target_col = cols[0]
        else:
            target_col = cols
        y_hat = model.predict(x_test)
        pred_df = Model.gen_pred_df(test_df, y_hat, target_col)
        base_score = Evaluator.eval_acc(pred_df)
        base_score

        num_samples = x_test.shape[0]

        scores = []
        for i in range(len(feature_names)):
            x_perm = x_test.copy()
            perm = np.random.permutation(np.array(range(num_samples)))
            x_perm[:,i] = x_test[perm,i]
            
            y_hat_perm = model.predict(x_perm)
            pred_df = Model.gen_pred_df(test_df, y_hat_perm, target_col)
            col_score = Evaluator.eval_acc(pred_df)
            scores.append(base_score-col_score)
        feature_df = pd.DataFrame({'features':feature_names, 'score':scores})
        feature_df = feature_df.sort_values('score',ascending=False)

        return feature_df