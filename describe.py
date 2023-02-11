import sys
import csv
import pandas as pd
import numpy as np

def mean(data):
    total = 0
    #nbr_element = len(data) I must check for the missing values
    nbr_element = 0
    for element in data:
        if ~np.isnan(element):
            total += element
            nbr_element+=1
    return round(total/nbr_element, 6)

def count(data):
    count = len(data)
    for element in data:
        if np.isnan(element):
            count-=1
    return count

def std(data):
    mean_val = mean(data)
    count = 0
    difference = 0
    for element in data:
        if ~np.isnan(element):
            difference += (element-mean_val) ** 2
            count+=1
    return round((difference / (count-1)) ** 0.5, 6)

def min(data):
    min = data[0]
    for element in data:
        if element < min:
            min = element
    return min

def max(data):
    max = data[0]
    for element in data:
        if element > max:
            max = element
    return max

def percentile(data, percent):
    data = [x for x in data if ~np.isnan(x)]
    # data.sort()
    # index = int(percent * (len(data) -1))
    # print(index)
    # return round(data[index], 6)
    return np.percentile(data, percent*100)

def describe_dataset(path):
    try:
        # Load the dataset into a Pandas Dataframe
        df = pd.read_csv(path)
      
        numerical_columns = df.select_dtypes(include=[np.number])
        mean_values     = {}
        max_values      = {}
        count_values    = {}
        min_values      = {}
        std_values      = {}
        first_quartile  = {}
        median          = {}
        third_quartile  = {}

        for col in numerical_columns:
            mean_values[col]    = mean(df[col])
            max_values[col]     = max(df[col])
            count_values[col]   = count(df[col])
            min_values[col]     = min(df[col])
            std_values[col]     = std(df[col])
            first_quartile[col] = percentile(df[col], 0.25)
            median[col]         = percentile(df[col], 0.50)
            third_quartile[col] = percentile(df[col], 0.75)
        #index = ['mean', 'count', 'min', 'max']
        data = {
            'Mean':pd.Series(mean_values),
            'Max':pd.Series(max_values),
            'Count':pd.Series(count_values),
            'Min':pd.Series(min_values),
            'std':pd.Series(std_values),
            '25%':pd.Series(first_quartile),
            '50%':pd.Series(median),
            '75%':pd.Series(third_quartile)
        }
        return pd.DataFrame(data)
        # print(pd.DataFrame(data))
        # print(df.describe())
    except Exception as e:
        print(f"Error : An exception occured {e}")
        sys.exit()


if __name__=='__main__':
    if len(sys.argv) != 2:
        print("Error : Invalid path or missing dataset. Please provide a valid path to the dataset and try again.")
        sys.exit()
    path = sys.argv[1]
    df = describe_dataset(path)
    print(df)