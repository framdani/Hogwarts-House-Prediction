import matplotlib.pyplot as plt
import numpy as np
from histogram import load_dataset, prepare_numerical_columns
import seaborn as sns
import argparse

def plot_scatter(df):
    #print(df['Hogwarts House'].value_counts())
    numerical_columns = prepare_numerical_columns(df)
    #Create a subplot for each feature
    n_features = numerical_columns.shape[1]
    ## To print all the plots into one figure
    n_rows = 11
    n_columns = 6
    counter = 0
   
    fig, axs = plt.subplots(n_rows, n_columns, figsize=(18,20))
    axs=axs.flatten()
    # print(axs.shape)
    for col1 in numerical_columns.columns:
        for col2 in numerical_columns.columns:
            if col1 != col2 and counter < 66:
                sns.scatterplot(x=numerical_columns[col1], y=numerical_columns[col2], hue=df['Hogwarts House'], ax = axs[counter])
                counter+=1
    # 
    plt.tight_layout()
    plt.show()

def fun(df,course1,course2):
    col1 = df[course1]
    col2 = df[course2]
    sns.scatterplot(data=df, x=col1, y = col2, hue='Hogwarts House')
    plt.xlabel(col1.name)
    plt.ylabel(col2.name)
    plt.title(f'Scatter plot of {course1} vs {course2}')
    plt.show()
   
if __name__=='__main__':
    try:
        parser = argparse.ArgumentParser('Plot scatter plots for pairwise combinations of features, coulored by house ')
        parser.add_argument('--plot_type', type=str, choices=['all_courses','two_courses'], default='all_courses', help='Type of plot to generate')
        parser.add_argument('--course1', type=str, default='Astronomy', help='Name of the first course')
        parser.add_argument('--course2', type=str, default='Defense Against the Dark Arts', help='Name of the second course')
        path = 'datasets/dataset_train.csv'
        df = load_dataset(path)

        args = parser.parse_args()
        if args.plot_type == 'all_courses':
            plot_scatter(df)
        else:
            if args.course1 is None or args.course2 is None:
                raise Exception(f"For plot type two_courses you must specify the names of two courses using the --course1 an --course2 args")
            else:
                fun(df,args.course1, args.course2)
    except Exception as e:
        print(f'An exception occured : {e}')
