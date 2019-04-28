import numpy as np
import pandas as pd

class Etl:
    def __init__(self, filename,data_set_name):
        self.filename = filename
        self.name = data_set_name
        self.exclude_cols = []
        self.target = None

        np.random.seed(1234)
    
    def train_test_split(self, df, split=.8):
        mask = np.random.rand(len(df)) < 0.8
        train_df = df[mask]
        test_df = df[mask]

        return train_df, test_df
    
    def remove_exclude_cols(self,df):
        if len(self.exclude_cols)==0:
            return df
        else:
            include_cols = [x for x in df.columns if x not in self.exclude_cols]
            return df[include_cols]

    def run(self):
        pass