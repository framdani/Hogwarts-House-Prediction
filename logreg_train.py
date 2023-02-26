import numpy as np
import pandas as pd
from logistic_regression import *
from sklearn.metrics import accuracy_score
import sys

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def train(df):

    # choose features
    X = df[['Charms','Flying','Divination', 'History of Magic','Ancient Runes','Astronomy','Herbology', 'Muggle Studies']] 
    y = df['Hogwarts House']

    # Create a one vs all logistic regression model
    OneVsAllmodel = OneVsAllLogisticRegression(0.00001, 100000)
   
    # Fit the model to the training data
    OneVsAllmodel.fit(X, y)
    
    #Save the weights to a file
    models  = OneVsAllmodel.models
    weights = [model.weights for model in models]
    biases  = [model.bias for model in models]
    with open("weights.csv", "w") as file:
        file.write("weights, biase\n")

        for i in range(len(models)):
            # Write the model number,weights, and bias to the file
            file.write(f"{','.join(map(str, weights[i]))}, {biases[i]}\n")


if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise Exception("Please provide one argument")
        path = sys.argv[1]
        df = load_dataset(path)
        df.replace('', np.nan, inplace=True)
        df.fillna(df.mean(), inplace=True)
        train(df)
    except Exception as e:
        print(f"An Exception occured : {e}")


    