from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class RfClass:
    def __init__(self, rf_params):
        self.rf_params = rf_params
        self.name = 'RF Classifier'

    def build_rf(self):
        max_depth = self.rf_params['max_depth']
        n_estimators = self.rf_params['n_estimators']

        rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

        return rf
    
    def split_xy(self, df, target_col):
        features = [x for x in df.columns if x != target_col]
        x = df[features].values
        y = df[target_col].values
        return x,y

    def run(self, train_df, test_df, target_col):
        rf = self.build_rf()

        x_train,y_train=self.split_xy(train_df,target_col)
        x_test,_=self.split_xy(test_df,target_col)

        rf.fit(x_train,y_train)

        y_test_hat = rf.predict(x_test)

        pred_df = test_df.copy()
        pred_df = pred_df.rename(columns={target_col: 'actual'})
        pred_df['pred'] = y_test_hat

        return pred_df

        
