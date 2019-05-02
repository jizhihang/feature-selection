import shap
import numpy as np
import pandas as pd

from feature_importance.feature_importance import FeatureImportance

class Shap(FeatureImportance):
    def __init__(self, model_type):
        super().__init__(model_type)
    
    @staticmethod
    def gen_shap_df(shap_values,feature_names):
        if len(shap_values.shape)==3:
            shap_values = np.mean(np.abs(shap_values), axis=1)
        else:
            shap_values = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame(data=shap_values,columns=feature_names)
        mean_shap_df = shap_df.mean().reset_index()
        mean_shap_df = mean_shap_df.rename(columns={'index':'features', 0:'value'})
        mean_shap_df = mean_shap_df.sort_values('value',ascending=False)
        return mean_shap_df

    def calc_rf(self, model, x_train, x_test, feature_names):
        explainer = shap.TreeExplainer(model)
        shap_values = np.array(explainer.shap_values(x_test))
        shap_df = Shap.gen_shap_df(shap_values,feature_names)
        return shap_df

    def calc_logreg(self, model, x_train, x_test, feature_names):
        explainer = shap.KernelExplainer(model.predict, x_train.astype(int), link='logit')
        shap_values = np.array(explainer.shap_values(x_test))
        shap_df = Shap.gen_shap_df(shap_values,feature_names)
        return shap_df
    
    def calc_nn(self, model, x_train, x_test, feature_names):
        background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
        explainer = shap.DeepExplainer(model, background)
        shap_values = np.array(explainer.shap_values(x_test))
        shap_df = Shap.gen_shap_df(shap_values,feature_names)
        return shap_df
    
    def calc_feature_importance(self, model, x_train, x_test, y_test, feature_names):
        if self.model_type=='rf_class':
            return self.calc_rf(model, x_train, x_test,feature_names)
        elif self.model_type=='log_reg':
            return self.calc_logreg(model,x_train,x_test,feature_names)
        elif self.model_type=='nn':
            return self.calc_nn(model,x_train,x_test,feature_names)
        else:
            return None