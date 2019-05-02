import pandas as pd

from etl.iris_etl import IrisEtl
from etl.zoo_etl import ZooEtl
from etl.credit_etl import CreditEtl

from models.rf_class import RfClass
from models.log_reg import LogReg
from models.neural_net import NeuralNet

from evaluators.evaluator import Evaluator

from feature_importance.shap import Shap
from feature_importance.permutations import Permutation

class FeatureSelectionDriver:
    def __init__(self,data_set,model_params,fimp_type):
        self.data_set = data_set
        self.model_params = model_params
        self.model_type = model_params['model_type']
        self.classification_task = True
        self.fimp_type = fimp_type

    def get_etl_pipeline(self):
        if self.data_set=='credit':
            etl = CreditEtl()
        elif self.data_set=='iris':
            etl =  IrisEtl()
        elif self.data_set=='zoo':
            etl = ZooEtl()
        else:
            return None
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
        return model
    
    def get_fimp_object(self):
        if self.fimp_type=='shap':
            fimp_object = Shap(self.model_type)
        elif self.fimp_type=='perm':
            fimp_object = Permutation(self.model_type)
        else:
            return None
        
        return fimp_object
    
    def class_pipeline(self,acc_percent_threshold,verbose):
        # ETL
        etl = self.get_etl_pipeline()
        if verbose:
            print("Using The " +etl.name+ " Data Set")
        one_hot_target = (self.model_type=='nn')
        train_df,val_df,test_df = etl.run(one_hot_target=one_hot_target,val_set=True)
        
        if isinstance(etl.target,list):
            target_arr = etl.target
        else:
            target_arr = [etl.target]

        fimp_object = self.get_fimp_object()

        features = [x for x in train_df.columns if x not in target_arr]
        new_features = features
        best_test_acc = 0
        best_val_acc = 0
        prev_test_acc = 0
        val_acc = 0

        feature_df = pd.DataFrame()
        while val_acc>=best_val_acc*acc_percent_threshold:
            num_features = len(new_features)
            num_targets = len(target_arr)
            
            # Construct Train & Test DFs
            cols = new_features+target_arr
            train_df = train_df[cols]
            val_df = val_df[cols]
            test_df = test_df[cols]

            # Model Training
            model = self.get_model(num_features,num_targets)
            if verbose:
                print("Training On "+model.name)
            pred_test_df,pred_val_df,feature_import_df = model.run(train_df,test_df,etl.target, val_df=val_df, fimp_object=fimp_object,verbose=verbose)

            # Model Evaluation
            val_acc = Evaluator.eval_acc(pred_val_df)
            test_acc = Evaluator.eval_acc(pred_test_df)
            best_val_acc = max(val_acc, best_val_acc)
            best_test_acc = max(test_acc, best_test_acc)

            cont_feature_selection = val_acc>=best_val_acc*acc_percent_threshold
            
            if verbose:
                print('Model With Features {} \nHas Test Accuracy Of {}% (Best={}%) - Continue={}\n'.format(new_features, test_acc*100, best_test_acc*100, cont_feature_selection))
            
            feature_iter_df = pd.DataFrame({'features':[new_features], 'num_features':num_features, 'val_acc':val_acc, 'test_acc':test_acc, 'continue':cont_feature_selection})
            feature_df = pd.concat([feature_df,feature_iter_df], sort=False)
            if cont_feature_selection:
                features=new_features
                prev_test_acc = test_acc
                new_features = feature_import_df[:-1]['features'].values.tolist()
                if len(new_features)==0:
                    break
        return feature_df

    
    def run(self,acc_percent_threshold,verbose=True):
        if self.classification_task:
            return self.class_pipeline(acc_percent_threshold,verbose)
        else:
            pass
            