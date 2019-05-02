import numpy as np
import pandas as pd

class Model:
    def __init__(self,model_params,name):
        self.model_params = model_params
        self.model_type=model_params['model_type']
        self.name = name
        self.features = None
    
    def build_model(self):
        return None

    def split_xy(self, df, target_col):
        if isinstance(target_col,list):
            target_arr = target_col
        else:
            target_arr = [target_col]
        self.features = [x for x in df.columns if x not in target_arr]
        x = df[self.features].values
        y = df[target_col].values
        return x,y

    def fit(self, model, x_train, y_train,verbose=False):
        model.fit(x_train,y_train)
        return model
    
    def predict(self,model,x):
        y_hat = model.predict(x)
        return y_hat

    @staticmethod
    def dummies_to_categorical(target_df):
        return target_df.columns[target_df.values.argmax(1)]

    @staticmethod
    def gen_pred_df(test_df, y_test_hat, target_col):
        pred_df = test_df.copy()
        if isinstance(target_col,list):
            target_df = pred_df[target_col]
            pred_df['actual'] = Model.dummies_to_categorical(target_df)
            y_hat_df = pd.DataFrame(data=y_test_hat, columns=target_col)
            pred_df['pred'] = Model.dummies_to_categorical(y_hat_df)
        else:
            pred_df = pred_df.rename(columns={target_col: 'actual'})
            pred_df['pred'] = y_test_hat
        
        return pred_df

    def get_feature_names(self, train_df, target_col):
        if not isinstance(target_col,list):
            target_col = [target_col]
        features = [x for x in train_df.columns if x not in target_col]
        return features

    def run(self, train_df, test_df, target_col, val_df=None, fimp_object=None, verbose=True):
        model = self.build_model()

        features = self.get_feature_names(train_df, target_col)

        x_train,y_train=self.split_xy(train_df,target_col)
        x_test,y_test=self.split_xy(test_df,target_col)
        if val_df is not None:
            x_val,y_val=self.split_xy(val_df,target_col)

        model = self.fit(model,x_train,y_train,verbose=verbose)

        y_test_hat = self.predict(model,x_test)
        pred_test_df = Model.gen_pred_df(test_df, y_test_hat, target_col)
        if val_df is not None:
            y_val_hat = self.predict(model,x_val)
            pred_val_df = Model.gen_pred_df(val_df, y_val_hat, target_col)

        if fimp_object != None:
            if verbose:
                print('Calculating Feature Importance')
            if len(x_train>100):
                x_train_fimp = x_train[np.random.randint(x_train.shape[0], size=100), :]
                feature_df = fimp_object.calc_feature_importance(model,x_train_fimp,x_test,y_test,features)
            else:
                feature_df = fimp_object.calc_feature_importance(model,x_train,x_test,y_test,features)
            
            if val_df is not None:
                return pred_test_df,pred_val_df,feature_df
            else:
                return pred_test_df,feature_df

        if val_df is not None:
            return pred_test_df, pred_val_df 
        else:
            return pred_test_df 