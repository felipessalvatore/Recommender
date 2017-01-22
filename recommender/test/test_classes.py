#!/usr/bin/env python3
from os import path
import numpy as np
import pandas as pd

import sys
parent_path = path.abspath('..')
sys.path.insert(0, parent_path)
import unittest
import dfFunctions
import tf_models
import recommender as re
from utils import rmse


def run_test(testClass, header):
    """
    Function to run all the tests from a class of tests.

    :type testClass: unittest.TesCase
    :type header: str
    """
    print(header)
    suite = unittest.TestLoader().loadTestsFromTestCase(testClass)
    unittest.TextTestRunner(verbosity=2).run(suite)


class TestBasic(unittest.TestCase):
    """
    Class that test all the basic functions
    """
    def test_rmse(self):
        """
        Test to check if the rmse function deals
        with arrays of different sizes and if it behaves
        normally.
        """
        array1 = np.array([1, 1, 1, 1])
        array2 = np.array([1, 1, 1, 1, 2])
        array3 = np.array([2, 2, 2, 2])
        self.assertRaises(AssertionError, rmse, array1, array2)
        self.assertTrue(rmse(array3, array1) == 1)


class TestdfManipulation(unittest.TestCase):
    """
    Class with tests for dataframe manipulation.
    """
    def test_load_dataframe(self):
        """
        Test to check if the function load_dataframe is working
        with all the datasets from movielens.
        """
        path1 = parent_path + '/movielens/ml-1m/ratings.dat'
        path10 = parent_path + '/movielens/ml-10m/ratings.dat'
        path20 = parent_path + '/movielens/ml-20m/ratings.csv'
        df1 = dfFunctions.load_dataframe(path1)
        df10 = dfFunctions.load_dataframe(path10)
        df20 = dfFunctions.load_dataframe(path20)
        self.assertTrue(type(df1) == pd.core.frame.DataFrame)
        self.assertTrue(type(df10) == pd.core.frame.DataFrame)
        self.assertTrue(type(df20) == pd.core.frame.DataFrame)

    def test_batch(self):
        """
        Test to check if the batchgenerator class is creating
        different batches of the same size.
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        batch_size = 100
        generator = dfFunctions.BatchGenerator(df,
                                               batch_size,
                                               'user',
                                               'item',
                                               'rating')
        old_observation = None
        count = 0
        num_of_tests = 200
        for i in range(num_of_tests):
            batch = generator.get_batch()
            current_observation = (batch[0][0], batch[1][0], batch[2][0])
            if current_observation == old_observation:
                count += 1
            old_observation = current_observation
            self.assertTrue(len(batch[0]) == batch_size)
            self.assertTrue(len(batch[1]) == batch_size)
            self.assertTrue(len(batch[2]) == batch_size)
        self.assertTrue(count < num_of_tests)

    def test_dataframe_separation(self):
        """
        Test to check if the class SVDmodel is separating
        the dataframe in train, test and valid dataframes
        with the right proportions.
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating')
        sum_of_sizes = len(model.train) + len(model.test) + len(model.valid)
        proportion_train = len(model.train)/len(df)
        proportion_test = len(model.test)/len(df)
        proportion_valid = len(model.valid)/len(df)
        right_proportions = np.array([0.8, 0.1, 0.1])
        proportions = np.array([proportion_train,
                                proportion_test,
                                proportion_valid])

        error = rmse(proportions, right_proportions)
        self.assertTrue(len(df) == sum_of_sizes)
        self.assertTrue(error < 0.1,
                        """\n The right proportions are (train,test,valid) =
                        {0}, but the model is separating the dataframe with the
                        proportions {1}"""
                        .format(right_proportions, proportions))

    def test_dataframe_intersection(self):
        """
        Test to check if the train, test and valid dataframes
        have no intersection between them.
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating')
        dic_intersection = dfFunctions.count_intersection(model.train,
                                                          model.test,
                                                          model.valid)
        self.assertTrue(dic_intersection['1-2'] == 0,
                        """\n The intersection between
            the train and test dataframe
            is {0}""".format(dic_intersection['1-2']))
        self.assertTrue(dic_intersection['1-3'] == 0,
                        """\n The intersection between
            the train and valid dataframe
            is {0}""".format(dic_intersection['1-3']))
        self.assertTrue(dic_intersection['2-3'] == 0,
                        """\n The intersection between
            the test and valid dataframe
            is {0}""".format(dic_intersection['2-3']))


class TestSVD(unittest.TestCase):
    """
    Class with tests for the SVD model.
    """
    def test_upperbound(self):
        """
        We run 5000 steps of training and check if the root mean square error
        from the valid dataset is less than 1.0 in the SVD model
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating')

        dimension = 12
        regularizer_constant = 0.05
        learning_rate = 0.001
        momentum_factor = 0.9
        batch_size = 1000
        num_steps = 5000

        print("\n")
        model.training(dimension,
                       regularizer_constant,
                       learning_rate,
                       momentum_factor,
                       batch_size,
                       num_steps)

        prediction = model.valid_prediction()
        self.assertTrue(prediction <= 1.0,
                        """\n with num_steps = {0} \n, the mean square error
                        of the valid dataset should be
                        less than 1 and not {1}"""
                        .format(num_steps, prediction))

    def test_prediction(self):
        """
        We run 5000 steps of training and check if the difference
        between the prediction mean and the actual mean is less than
        0.9 in the SVD model
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating')

        dimension = 12
        regularizer_constant = 0.05
        learning_rate = 0.001
        momentum_factor = 0.9
        batch_size = 1000
        num_steps = 5000

        print("\n")
        model.training(dimension,
                       regularizer_constant,
                       learning_rate,
                       momentum_factor,
                       batch_size,
                       num_steps)

        user_example = np.array(model.valid['user'])[0:10]
        movies_example = np.array(model.valid['item'])[0:10]
        actual_ratings = np.mean(np.array(model.valid['rating'])[0:10])
        predicted_ratings = np.mean(model.prediction(user_example,
                                                     movies_example))
        difference = np.absolute(actual_ratings - predicted_ratings)

        self.assertTrue(difference <= 0.9,
                        """\n with num_steps = {0} \n, the difference should be
                        less than 0.9 and not {1}"""
                        .format(num_steps, difference))


class TestNSVD(unittest.TestCase):
    """
    Class with tests for the NSVD model.
    """
    def test_rated_items(self):
        """
        Test to check if the method _set_item_dic creates
        a dic with user:rated_items such that
        len(rated_items) == max_size.
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        finder = dfFunctions.ItemFinder(df, 'user', 'item', 'rating', 'mean')
        all_users = df['user'].unique()
        count = 0
        problem_users = []
        for user in all_users:
            r_items = finder.dic[user]
            if len(r_items) == finder.size and r_items.dtype == 'int32':
                count += 1
            else:
                problem_users.append(user)
        self.assertTrue(count == len(all_users),
                        """\n There are {0} arrays in dic such that
            len(dic[user]) != finder.size or with wrong types. And these
            users are {1}""".format(count, problem_users))

    def test_upperbound(self):
        """
        We run 5000 steps of training and check if the root mean square error
        from the valid dataset is less than 1.0 in the NSVD model
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating', 'nsvd', 'mean')

        dimension = 12
        regularizer_constant = 0.05
        learning_rate = 0.001
        momentum_factor = 0.9
        batch_size = 1000
        num_steps = 5000

        print("\n")
        model.training(dimension,
                       regularizer_constant,
                       learning_rate,
                       momentum_factor,
                       batch_size,
                       num_steps)

        prediction = model.valid_prediction()
        self.assertTrue(prediction <= 1.0,
                        """\n with num_steps = {0} \n, the mean square
                        error of the valid dataset should be less
                        than 1 and not {1}"""
                        .format(num_steps, prediction))

    def test_prediction(self):
        """
        We run 5000 steps of training and check if the difference
        between the prediction mean and the actual mean is less than
        0.9 in the SVD model
        """
        path = parent_path + '/movielens/ml-1m/ratings.dat'
        df = dfFunctions.load_dataframe(path)
        model = re.SVDmodel(df, 'user', 'item', 'rating', 'nsvd', 'mean')

        dimension = 12
        regularizer_constant = 0.05
        learning_rate = 0.001
        momentum_factor = 0.9
        batch_size = 1000
        num_steps = 5000

        print("\n")
        model.training(dimension,
                       regularizer_constant,
                       learning_rate,
                       momentum_factor,
                       batch_size,
                       num_steps)

        user_example = np.array(model.valid['user'])[0:10]
        movies_example = np.array(model.valid['item'])[0:10]
        actual_ratings = np.mean(np.array(model.valid['rating'])[0:10])
        predicted_ratings = np.mean(model.prediction(user_example,
                                                     movies_example))
        difference = np.absolute(actual_ratings - predicted_ratings)

        self.assertTrue(difference <= 0.9,
                        """\n with num_steps = {0} \n, the difference should be
                        less than 0.9 and not {1}"""
                        .format(num_steps, difference))
