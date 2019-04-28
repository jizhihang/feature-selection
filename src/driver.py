from etl.iris_etl import IrisEtl
from etl.zoo_etl import ZooEtl
from etl.credit_etl import CreditEtl

from models.rf_class import RfClass

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
    
    def get_model(self):
        if self.model_type=='rf_class':
            model=RfClass(self.model_params)
            print("Training On "+model.name)
            return model
        else:
            return None

    def class_pipeline(self):
        # ETL
        etl = self.get_etl_pipeline()
        train_df,test_df = etl.run()

        # Model Training
        model = self.get_model()
        pred_df = model.run(train_df,test_df,etl.target)

        # Model Evaluation
        acc = Evaluator.eval_acc(pred_df)
        print('Model Has Test Accuracy Of {}%'.format(acc*100))

    
    def run(self):
        if self.classification_task:
            self.class_pipeline()
        else:
            pass
            