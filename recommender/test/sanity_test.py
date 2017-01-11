#!/usr/bin/env python3
from os import getcwd,path, pardir
import numpy as np

import sys
pwd = getcwd()
parent_path = path.abspath(path.join(pwd, pardir)) 
sys.path.insert(0, parent_path)
import unittest
import dfFunctions
import tf_models
import recommender as re



class TestRecomendation(unittest.TestCase):

     def test_accuracy(self):
        """
        Test to check if the accuracy functions deals
        with arrays of different sizes and if it behaves
        normally.
        """
        array1 = np.array([1,1,1,1])
        array2 = np.array([1,1,1,1,2])
        array3 = np.array([2,2,2,2])
        self.assertRaises(accuracy(array1,array2))
        self.assertTrue(accuracy(array2,array1) == 1)   


    def test_upperbound(self):
        """
        A very basic test to check if the otimization is working.
        We run 5000 steps of training and check if the mean square error
        from the valid dataset is less than 1
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.get_data(path, sep="::")
        model = re.SVDmodel(df,'user', 'item','rate')

        dimension = 15
        regularizer_constant = 0.05
        learning_rate = 0.001
        batch_size = 1000
        num_steps = 5000

        model.training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
        prediction = model.valid_prediction()
        self.assertTrue(prediction <=1.0, \
                            "\n with num_steps = {0} \n, the mean square error of the valid dataset should be less than 1 and not {1}"\
                            .format(num_steps,prediction))





def run_test():
    """ 
    Running all the tests. This code should have a more 
    robust test framework.
    """
    print("Running all tests...")
    suite = unittest.TestSuite()
    for method in dir(TestRecomendation):
       if method.startswith("test"):
          suite.addTest(TestRecomendation(method))
    unittest.TextTestRunner().run(suite)


if __name__ == '__main__':
    run_test()

