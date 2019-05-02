import numpy as np
import pandas as pd

class Etl:
    def __init__(self, filename,data_set_name):
        self.filename = filename
        self.name = data_set_name
        self.exclude_cols = []
        self.target = None
        self.target_is_categorical = False

        np.random.seed(1234)
    
    def extract(self):
        return None

    def transform(self,df,one_hot_target=False,val_set=False):
        df = self.remove_exclude_cols(df)
        split_df = self.train_test_split(df,val_set=val_set)
        if one_hot_target:
            split_df = self.one_hot_target(split_df)
        cols = [x for x in split_df.columns if x not in ['train','val','test']]
        if val_set:
            train_df = split_df[split_df['train']==True][cols]
            val_df = split_df[split_df['val']==True][cols]
            test_df = split_df[split_df['test']==True][cols]
            return train_df,val_df,test_df
        else:
            train_df = split_df[split_df['train']==True][cols]
            test_df = split_df[split_df['test']==True][cols]
            return train_df,test_df

    def train_test_split(self, df, split=.8, val_set=False):
        split_df = pd.DataFrame()
        for target in df[self.target].unique():
            dft = df[df[self.target]==target].copy()
            r = np.random.rand(len(dft))
            mask = r < 0.8
            dft['train']=mask
            if val_set:
                mask_val = r>.9
                dft['test'] = (~mask)&mask_val
                dft['val'] = (~mask)&(~mask_val)
            else:
                dft['test'] = ~mask
            split_df = pd.concat([split_df,dft])
        return split_df
    
    def remove_exclude_cols(self,df):
        if len(self.exclude_cols)==0:
            return df
        else:
            include_cols = [x for x in df.columns if x not in self.exclude_cols]
            return df[include_cols]
    
    def one_hot_target(self,df):
        target_df=pd.get_dummies(df[self.target])
        if ~isinstance(target_df,str):
            target_df.columns = [str(x) for x in target_df.columns]
        features = [x for x in df.columns if x!=self.target]
        df=pd.concat([df[features],target_df],axis=1)
        self.target = target_df.columns.values.tolist()
        return df

    def run(self, one_hot_target=False, val_set=False):
        df = self.extract()
        return self.transform(df, one_hot_target=one_hot_target, val_set=val_set)