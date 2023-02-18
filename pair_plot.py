import matplotlib.pyplot as plt
import numpy as np
from histogram import load_dataset, prepare_numerical_columns
import seaborn as sns


def pair_plot(df):
    sns.pairplot(df, diag_kind='hist', hue='Hogwarts House')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    path = 'datasets/dataset_train.csv'
    df = load_dataset(path)
    pair_plot(df)