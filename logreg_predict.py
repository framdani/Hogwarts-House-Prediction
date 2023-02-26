import numpy as np
import pandas as pd
import csv
from logistic_regression import OneVsAllLogisticRegression
import sys

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def predict(X_test,weights, labels):
    preds = []
    for i in range(weights.shape[0]):
        z = np.dot(X_test, weights[i][:-1]) + weights[i][-1]
        y_hat = 1/(1+np.exp(-z))
        preds.append(y_hat)
    preds = np.array(preds).T
    return labels[np.argmax(preds, axis=1)]


if __name__ == '__main__':
    try:
        
        if len(sys.argv) != 3:
            raise Exception("Please provide two argument")
        
        # Define labels
        df = load_dataset('datasets/dataset_train.csv')
        labels = np.unique(df['Hogwarts House'])

        # load the test dataset
        path = sys.argv[1]
        df_test = load_dataset(path)
        df_test.replace('', np.nan, inplace=True)
        df_test.fillna(df.mean(), inplace=True)
        X_test = df_test[['Charms','Flying','Divination', 'History of Magic','Ancient Runes','Astronomy','Herbology', 'Muggle Studies']] 
        
        # load the weights and biases
        weights = np.zeros((4, 9))
        weights_path = sys.argv[2]
        with open(weights_path, "r") as file:
            reader = csv.reader(file, delimiter = ',')
            next(reader)
            for i, row in enumerate(reader):
                for j, value in enumerate(row):
                    weights[i, j] = value
        
        result = predict(X_test, weights, labels)
        df = pd.DataFrame({'Hogwarts House':result})
        df.index.name = 'Index'
        df.to_csv('houses.csv')
    except Exception as e:
        print(f"An Exception occured : {e}")