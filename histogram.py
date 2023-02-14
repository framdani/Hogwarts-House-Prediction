
# Determine Which Hogwarts course has a homogenous score distribution between all four houses
# Investigate if the distribution of scores is alike among the different groups; the four houses in Hogwarts
# Answer : Arithmacy | Care of magical creatures

import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def prepare_numerical_columns(df):
    numerical_columns = df.select_dtypes(include=[np.number])
    numerical_columns = numerical_columns.drop(columns=['Index'])
    return numerical_columns

def plot_histogram(df):
    numerical_columns = prepare_numerical_columns(df)
    groupes = df.groupby("Hogwarts House")
    courses = numerical_columns.columns
    n_courses= len(courses)
    n_rows = int(np.ceil(n_courses/4))
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15), tight_layout=True)
    
    #enumerate can only loop over a 1D array
    axs = axs.ravel()
    for i, course in enumerate(courses):
        for j, (name, group) in enumerate(groupes):
            axs[i].hist(group[course], bins=20, alpha = 0.6, label=name)
            axs[i].set_title(course)
            axs[i].legend()
    plt.show()


if __name__=='__main__':
    path = 'datasets/dataset_train.csv'
    df = load_dataset(path)
    plot_histogram(df)