#!/usr/bin/env python3

import numpy as np
import dfFunctions
import tf_models


class SVDmodel(object):
    """
    Class to creat SVD models and NSVD models. This class does not deal with
    tensorflow. It separate the dataframe in three parts: train, test and
    validation; with that it comunicates with the class tf_models.SVD to
    creat a training session and to create a prediction.

    We use the params users, items and ratings to get the names
    from the columns of df.


    :type df: dataframe
    :type users: string
    :type items: string
    :type ratings: string
    :type model: string
    :type nsvd_size: string
    """
    def __init__(self,
                 df,
                 users,
                 items,
                 ratings,
                 model='svd',
                 nsvd_size='mean'):
        self.df = df
        self.users = users
        self.items = items
        self.ratings = ratings
        self.model = model
        self.size = len(df)
        self.num_of_users = max(self.df[self.users]) + 1
        self.num_of_items = max(self.df[self.items]) + 1
        self.train, self.test, self.valid = self.data_separation()
        if model == 'nsvd':
            self.finder = dfFunctions.ItemFinder(df, self.users,
                                                 self.items,
                                                 self.ratings, nsvd_size)

    def data_separation(self):
        """
        Method that randomizes the dataframe df and separate it
        in tree parts: 80% in traing, 10% in test and 10% in validation.

        :rtype: triple of dataframes
        """
        rows = len(self.df)
        random_ids = np.random.permutation(rows)
        random_df = self.df.iloc[random_ids].reset_index(drop=True)
        split_index = int(rows * 0.8)
        new_split = split_index + int((rows - split_index) * 0.5)
        df_train = random_df[0:split_index]
        df_test = random_df[split_index: new_split].reset_index(drop=True)
        df_validation = random_df[new_split:].reset_index(drop=True)
        return df_train, df_test, df_validation

    def training(self,
                 hp_dim,
                 hp_reg,
                 learning_rate,
                 momentum_factor,
                 batch_size,
                 num_steps,
                 verbose=True):
        """
        This method creates three batch generators: one for the train df,
        other for the test df, and another for the valid df (this last one will
        creats batchs of the size of the whole valid df); and it also creats
        one object tf_models.SVD (a kind of counterpart of the object SVDmodel
        that works with tensorflow) and request one training to tf_models.SVD.
        The object tf_models.SVD is save as self.tf_counterpart for the
        prediction. To print all training information you can set verbose=True,
        otherwise use verbose=False.

        :type hp_dim: int
        :type hp_reg: float
        :type momentum_factor: float
        :type learning_rate: float
        :type batch_size: int
        :type num_steps: int
        :type verbose: boolean
        """
        self.train_batches = dfFunctions.BatchGenerator(self.train,
                                                        batch_size,
                                                        self.users,
                                                        self.items,
                                                        self.ratings)

        self.test_batches = dfFunctions.BatchGenerator(self.test,
                                                       batch_size,
                                                       self.users,
                                                       self.items,
                                                       self.ratings)

        self.valid_batches = dfFunctions.BatchGenerator(self.valid,
                                                        len(self.valid),
                                                        self.users,
                                                        self.items,
                                                        self.ratings)
        if self.model == 'svd':
            self.tf_counterpart = tf_models.SVD(self.num_of_users,
                                                self.num_of_items,
                                                self.train_batches,
                                                self.test_batches,
                                                self.valid_batches)
        else:
            self.tf_counterpart = tf_models.SVD(self.num_of_users,
                                                self.num_of_items,
                                                self.train_batches,
                                                self.test_batches,
                                                self.valid_batches,
                                                self.finder,
                                                self.model)

        self.tf_counterpart.training(hp_dim,
                                     hp_reg,
                                     learning_rate,
                                     momentum_factor,
                                     num_steps,
                                     verbose)
        self.duration = round(self.tf_counterpart.general_duration, 2)
        if verbose:
            self.tf_counterpart.print_stats()

    def valid_prediction(self):
        """
        This method calls the tf_models.SVD and returns the mean
        square error of the whole valid dataset.

        :rtype: float
        """
        return self.tf_counterpart.prediction(show_valid=True)

    def prediction(self, list_of_users, list_of_items):
        """
        Given one np.array of users and one np.array of items,
        this method calls the tf_models.SVD and returns one np.array
        of predicted ratings.

        :rtype: numpy array of floats
        """
        return self.tf_counterpart.prediction(list_of_users, list_of_items)
