# What are the two features that are similar ?
# Answer Astronomy and Defense against the dark arts

import matplotlib.pyplot as plt
import numpy as np
from histogram import load_dataset, prepare_numerical_columns

def plot_scatter(df):
    # numerical_columns = prepare_numerical_columns(df)
    # Create a subplot for each feature
    # n_features = numerical_columns.shape[1]
    ### To print all the plots into one figure
    # n_rows = 11
    # n_columns = 6
    # counter = 0
   
    # fig, axs = plt.subplots(n_rows, n_columns, figsize=(18,20))
    # axs=axs.flatten()
    # print(axs.shape)
    # for col1 in numerical_columns.columns:
    #     for col2 in numerical_columns.columns:
    #         if col1 != col2 and counter < 66:
    #             axs[counter].scatter(numerical_columns[col1], numerical_columns[col2])
    #             axs[counter].set_xlabel(col1)
    #             axs[counter].set_ylabel(col2)
    #             counter+=1
    col1 = df['Astronomy']
    col2 = df['Defense Against the Dark Arts']
    plt.scatter(col1, col2)
    plt.xlabel(col1.name)
    plt.ylabel(col2.name)
    plt.title('Scatter plot of Astronomy vs Defense Against the Dark Arts')
    #plt.tight_layout()
    plt.show() 
   
if __name__=='__main__':
    path = 'datasets/dataset_train.csv'
    df = load_dataset(path)
    plot_scatter(df)
