import pandas as pd
from etl.etl import Etl

class CreditEtl(Etl):
    def __init__(self, filename='../data/credit/default of credit card clients.xls'):
        super().__init__(filename,'Default of Credit Card Clients')
        self.exclude_cols = ['ID']
        self.target = 'default payment next month'
    
    def extract(self):
        df = pd.read_excel(self.filename, header=1, index=0)
        return df

    def transform(self,df):
        df = self.remove_exclude_cols(df)
        train_df,test_df = self.train_test_split(df)
        return train_df,test_df

    def run(self):
        df = self.extract()
        train_df,test_df = self.transform(df)
        return train_df,test_df