from etl.iris_etl import IrisEtl
from etl.zoo_etl import ZooEtl
from etl.credit_etl import CreditEtl

from models.rf_class import RfClass
from models.log_reg import LogReg
from models.neural_net import NeuralNet

from evaluators.evaluator import Evaluator

class Driver:
    def __init__(self,data_set,model_params):
        self.data_set = data_set
        self.model_params = model_params
        self.model_type = model_params['model_type']
        self.classification_task = True

    def get_etl_pipeline(self):
        if self.data_set=='credit':
            etl = CreditEtl()
        elif self.data_set=='iris':
            etl =  IrisEtl()
        elif self.data_set=='zoo':
            etl = ZooEtl()
        else:
            return None

        print("Using The " +etl.name+ " Data Set")
        return etl
    
    def get_model(self,num_features,num_targets):
        if self.model_type=='rf_class':
            model=RfClass(self.model_params)
        elif self.model_type=='log_reg':
            model=LogReg(self.model_params) 
        elif self.model_type=='nn':
            self.model_params['num_features']=num_features
            self.model_params['num_targets']=num_targets
            model=NeuralNet(self.model_params)
        else:
            return None
        print("Training On "+model.name)
        return model
    
    def class_pipeline(self):
        # ETL
        etl = self.get_etl_pipeline()
        one_hot_target = etl.target_is_categorical&(self.model_type=='nn')
        train_df,test_df = etl.run(one_hot_target=one_hot_target)
        
        if isinstance(etl.target,list):
            target_arr = etl.target
        else:
            target_arr = [etl.target]

        num_features = len([x for x in train_df.columns if x not in target_arr])
        num_targets = len(target_arr)

        # Model Training
        model = self.get_model(num_features,num_targets)
        pred_df,shap_df = model.run(train_df,test_df,etl.target, calc_shap=True)

        # Model Evaluation
        acc = Evaluator.eval_acc(pred_df)
        print('Model Has Test Accuracy Of {}%'.format(acc*100))

        return pred_df,shap_df

    
    def run(self):
        if self.classification_task:
            return self.class_pipeline()
        else:
            pass
            