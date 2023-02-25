import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01,epochs=1000, verbose=False):
        # Initializethe weights and bias
        # Define learning rate and number of epochs
        self.lr      = lr
        self.epochs  = epochs
        self.verbose = verbose
        self.weights = None
        self.bias    = None
        self.costs   = []
    
    def __sigmoid(self, z):
        """
        takes an input and returns an output between 0 and 1
        """
        return 1/(1+np.exp(-z))
    
    def __loss(self, X,y):
        """
        mesures the error of the model's prediction relative to the true labels
        """
        n_samples = len(y)
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.__sigmoid(z)
        cost = (-1/n_samples) * np.sum(y * np.log(y_hat)) + (1 - y) * np.log(1 - y_hat)
        return cost
    
    def fit(self, X,y):
        # Weights and bias initialization
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        n_samples = len(y)

        #Performing gradient descent
        for epoch in range(self.epochs):
            z = np.dot(X, self.weights) + self.bias
            y_hat = self.__sigmoid(z)

            dw = (1/n_samples) * np.dot(X.T, (y_hat - y))
            db = (1/n_samples) * np.sum(y_hat - y)
            #Update weights and bias
            self.weights -= self.lr * dw
            self.bias    -=self.lr * db
            #Save loss for each iteration
            self.costs.append(self.__loss(X, y))
    
    def predict_prob(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_hat = self.__sigmoid(z)
        return y_hat

    def predict(self, X, treshold=0.5):
        return self.predict_prob(X) >= treshold

class OneVsAllLogisticRegression:
    # split data into n classes
    # for each class a binary classification is created
    # for each binary classification, a logRegression model is trained
    #To classify a new input sample, the input is passed through all the binary problems
    # the highest prob is considered the predicted class
    # 
    def __init__(self, lr, epochs, verbose):
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.models = []
        self.labels = None

    def __split_labels(self, y):
        self.labels = np.unique(y)
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))
        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where( y == label)] =1
        return splitted_labels
    
    def fit(self, X, y):
        labels = self.__split_labels(y)
        for label in labels:
            model = LogisticRegression(lr=self.lr, epochs =self.epochs,verbose=self.verbose)
            model.fit(X, label)
            self.models.append(model)


    def predict(self, X):
        preds = []
        for model in self.models:
            preds.append(model.predict_prob(X))
        
        preds= np.array(preds).T
        return self.labels[np.argmax(preds, axis=1)]
