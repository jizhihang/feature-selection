import pandas as pd

class CreditEtl:
    def __init__(self, filename):
        self.filename = filename

        self.exclude_cols = ['ID']
        self.target = 'default payment next month'
    
    def run():
        df = pd.read_excel('../data/credit/default of credit card clients.xls', header=1, index=0)
        return df