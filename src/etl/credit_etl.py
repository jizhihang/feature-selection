import pandas as pd

# '../data/credit/default of credit card clients.xls'

class CreditEtl:
    def __init__(self, filename):
        self.filename = filename

        self.exclude_cols = ['ID']
        self.target = 'default payment next month'
    
    def run(self):
        df = pd.read_excel(self.filename, header=1, index=0)
        return df