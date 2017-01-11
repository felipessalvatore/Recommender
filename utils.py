import numpy as np

def accuracy(predictions, ratings):
    """
    Return the mean square error between the 
    predictions vector and the ratings vectors.


    :type predictions: numpy array 	
    :type ratings: numpy array
    :rtype: float
    """
    return np.sqrt(np.mean(np.power(predictions - ratings, 2)))
