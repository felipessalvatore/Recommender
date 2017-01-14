import numpy as np

def mse(x, y):
	"""
	Return the mean square error between the 
	x and y arrays.


	:type x: numpy array 	
	:type y: numpy array
	:rtype: float
	"""
	assert len(x) == len(y)
	return np.mean(np.power(x - y, 2))

def rmse(x,y):
    """
    Return the root-mean-square error between the 
    x and y arrays.


    :type x: numpy array 	
    :type y: numpy array
    :rtype: float
    """
    return np.sqrt(mse(x,y))