import pandas as pd
from etl.etl import Etl

class IrisEtl(Etl):
    def __init__(self, filename='../data/iris/iris.data'):
        super().__init__(filename,'Iris')
        
        """
        Column Information:

        1. sepal length in cm
        2. sepal width in cm
        3. petal length in cm
        4. petal width in cm
        5. class: 
        -- Iris Setosa
        -- Iris Versicolour
        -- Iris Virginica
        """
        self.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

        self.exclude_cols = []
        self.target = 'class'
        self.target_is_categorical = True
    
    def extract(self):
        df = pd.read_csv(self.filename, header=None)
        df.columns = self.columns
        return df
        