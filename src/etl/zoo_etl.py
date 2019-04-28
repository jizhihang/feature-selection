import pandas as pd
from etl.etl import Etl

class ZooEtl(Etl):
    def __init__(self, filename='../data/zoo/zoo.data'):
        super().__init__(filename, 'Zoo')

        """
        Column Information:

        1. animal name: Unique for each instance
        2. hair Boolean
        3. feathers Boolean
        4. eggs Boolean
        5. milk Boolean
        6. airborne Boolean
        7. aquatic Boolean
        8. predator Boolean
        9. toothed Boolean
        10. backbone Boolean
        11. breathes Boolean
        12. venomous Boolean
        13. fins Boolean
        14. legs Numeric (set of values: {0,2,4,5,6,8})
        15. tail Boolean
        16. domestic Boolean
        17. catsize Boolean
        18. type Numeric (integer values in range [1,7])
        """
        self.columns = ['name', 'hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 'preditor', 'toothed', 'backbone', 'breathes', 'venemous',
                'fins', 'legs', 'tail', 'domestic', 'catsize', 'type']

        self.exclude_cols = ['name']
        self.target = 'type'
    
    def extract(self):
        df = pd.read_csv(self.filename, header=None)
        df.columns = self.columns
        return df

    def transform(self,df):
        df = self.remove_exclude_cols(df)
        train_df,test_df = self.train_test_split(df)
        return train_df,test_df

    def run(self):
        df = self.extract()
        train_df,test_df = self.transform(df)
        return train_df,test_df