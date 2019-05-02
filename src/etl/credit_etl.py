import pandas as pd
from etl.etl import Etl

class CreditEtl(Etl):
    def __init__(self, filename='../data/credit/default of credit card clients.xls'):
        super().__init__(filename,'Default of Credit Card Clients')
        self.exclude_cols = ['ID']
        self.target = 'default payment next month'
        self.target_is_categorical = False
    
    def extract(self):
        df = pd.read_excel(self.filename, header=1, index=0)
        df = df.rename(columns={self.target:'default'})
        self.target='default'
        return df