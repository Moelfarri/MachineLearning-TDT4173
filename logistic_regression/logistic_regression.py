import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class LogisticRegression:
    
    def __init__(self, lr=0.01,iterations=1000):
        """Implementing Logistic Regression
        with SGD as an optimizer - skleran uses SAG.
        """
        self.lr         = lr
        self.iterations = iterations
        self.weights    = None
        self.bias       = 0
        
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        m_samples, n_features = np.shape(X)
        self.weights          = np.zeros(n_features) #init weights as zeros
      
        
        
        for i in range(self.iterations):
            #forward
            f = X@self.weights + np.array(self.bias) #w*x + b
            y_pred = sigmoid(f)
              
            assert y.shape == y_pred.shape
            
            #minimize cost function with respect to weights and bias
            dw = -(1/n_features)*X.T@(y-y_pred) 
            db = -(1/n_features)*np.sum(y-y_pred)
            
            #Stochastic Gradient Descent (backstepping)
            self.weights = gradient_step(self.weights, dw, self.lr)
            self.bias    = gradient_step(self.bias, db, self.lr)


    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats in the range [0, 1]
            with probability-like predictions
        """
        f = X@self.weights + np.array(self.bias) 
        y_pred = sigmoid(f)
        return y_pred

        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))


def gradient_step(x, dx, lr):
    return x - lr*dx