import pandas as pd

class ZooEtl:
    def __init__(self, filename):
        self.filename = filename

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
    
    def run():
        df = pd.read_csv('../data/zoo/zoo.data', header=None)
        df.columns = self.columns
        return df