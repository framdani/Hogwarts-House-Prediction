import pandas as pd
import helpers
import sys
import numbers
import numpy

class Describer():
    def __init__(self, csv_file : str) -> None :
        self.df = pd.read_csv(csv_file, index_col=0)
        if self.df is None or self.df.empty:
            raise ValueError('Empty DataFrame')
    
    def exclude_nan_columns(self) -> pd.DataFrame :
        return self.df.select_dtypes(include=[numpy.number])
    
    def describe(self) -> pd.DataFrame :
        data = {}
        for col_name, col_data in self.df.iteritems():
            col_data = helpers.manual_sort(col_data)
            data[col_name] = {}
            data[col_name]['count'] = helpers.count(col_data)
            data[col_name]['mean'] = helpers.mean(col_data)
            data[col_name]['std'] = helpers.std(col_data)
            data[col_name]['min'] = helpers.min(col_data)
            data[col_name]['25%'] = helpers.percentile(col_data, 25)
            data[col_name]['50%'] = helpers.percentile(col_data, 50)
            data[col_name]['75%'] = helpers.percentile(col_data, 75)
            data[col_name]['max'] = helpers.max(col_data)

        describe_df = pd.DataFrame.from_dict(data, orient='columns')\
                .applymap(lambda x: x if x != 0 else "{:.1f}".format(x))\
                .applymap(lambda x: x if not isinstance(x, numbers.Number) else "{:.06f}".format(x))
        return describe_df

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Error : Invalid number of arguments')
        exit(1)
    try:
        describer = Describer(sys.argv[1])
        describer.df = describer.exclude_nan_columns()
        print(describer.describe())
    except ValueError as e:
        print('Error : ' + str(e))
        exit(1)
    

    # print(describer.df.describe())

    # print('equals' if describer.df.describe().applymap(lambda x: x if x != 0 else "{:.1f}".format(x))\
    #         .applymap(lambda x: x if not isinstance(x, numbers.Number) else "{:.06f}".format(x)).equals(describer.describe()) else 'not equals')
