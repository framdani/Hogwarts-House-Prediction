# Look for features that look strongly associated with a particular house
# if for example data point of a particular house is clustered in one area of the plot

import matplotlib.pyplot as plt
import numpy as np
from histogram import load_dataset, prepare_numerical_columns
import seaborn as sns


def pair_plot(df):
    courses = ['Arithmancy', 'Astronomy', 'Herbology', 'Divination', 'Muggle Studies'
                , 'Ancient Runes', 'History of Magic', 'Transfiguration', 'Potions', 'Charms', 'Flying', 'Hogwarts House']
    sns.pairplot(df[courses], hue='Hogwarts House')
    plt.tight_layout()
    plt.show()

if __name__=='__main__':
    path = 'datasets/dataset_train.csv'
    df = load_dataset(path)
    pair_plot(df)