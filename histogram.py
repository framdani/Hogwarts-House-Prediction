import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import argparse

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def prepare_numerical_columns(df):
    numerical_columns = df.select_dtypes(include=[np.number])
    numerical_columns = numerical_columns.drop(columns=['Index'])
    return numerical_columns

def plot_histograms_for_all_courses(df):
    numerical_columns = prepare_numerical_columns(df)
    groupes = df.groupby("Hogwarts House")
    courses = numerical_columns.columns
    n_courses= len(courses)
    n_rows = int(np.ceil(n_courses/4))
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15), tight_layout=True)
    axs = axs.ravel()
    for i, course in enumerate(courses):
        for j, (name, group) in enumerate(groupes):
            axs[i].hist(group[course], bins=20, alpha = 0.6, label=name)
            axs[i].set_title(course)
            axs[i].legend()
    plt.show()

def plot_course_histogram_by_house(df, course):
    valid_cols= prepare_numerical_columns(df).columns
    if course not in valid_cols:
        raise Exception(f'{course} is not a valid course name.')

    Hufflepuff  = df[df['Hogwarts House'] == 'Hufflepuff'][course]
    Ravenclaw   = df[df['Hogwarts House'] == 'Ravenclaw'][course]  
    Gryffindor  = df[df['Hogwarts House'] == 'Gryffindor'][course]
    Slytherin   = df[df['Hogwarts House'] == 'Slytherin'][course]

    n_rows = 1
    n_cols = 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 15), tight_layout=True)
    axs = axs.ravel()
    houses = ['Hufflepuff', 'Ravenclaw', 'Gryffindor', 'Slytherin']
    for i, house in enumerate(houses):
        data=eval(house)
        axs[i].hist(data,bins=10, alpha=0.9, color='Green')
        axs[i].set_title(house)
    
    plt.xlabel(course)
    plt.ylabel('count')
    plt.suptitle(f'Scores of {course} scores by House')
    plt.show()

if __name__=='__main__':
    try:
        path = 'datasets/dataset_train.csv'
        df = load_dataset(path)
        parser = argparse.ArgumentParser(description='Plot histograms for Hogwarts courses by house')
        parser.add_argument('--course',type=str, default=None, help='Name of the course to plot histogram for')
        args=parser.parse_args()
        
        if args.course:
            plot_course_histogram_by_house(df, args.course)
        else:
            plot_histograms_for_all_courses(df)
            
    except Exception as e:
        print(f"An exception occured : {e} ")
