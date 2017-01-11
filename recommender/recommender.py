#!/usr/bin/env python3

import numpy as np
import dfFunctions
import tf_models
from os import getcwd


class SVDmodel(object):
    """ 
    Class to creat SVD models. This class does not deal with tensorflow. It
    separate the dataframe in three parts: train, test and validation; with
    that it comunicates with the class tf_models.SVD to creat a training 
    session and to create a prediction. 

    We use the params users, items and ratings to get the names
    from the columns of df.


    :type df: dataframe  
    :type users: string  
    :type items: string
    :type ratings: string
    """
    def __init__(self,df,users, items, ratings):
        self.df = df
        self.users = users
        self.items = items
        self.ratings = ratings
        self.size = len(df)
        self.num_of_users = max(self.df[self.users]) + 1
        self.num_of_items = max(self.df[self.items]) + 1
        self.train,self.test,self.valid = self.data_separation()
        
    def data_separation(self):
        """ 
        Method that randomizes the dataframe df and separate it
        in tree parts: 80% in traing, 10% in test and 10% in validation.
        
        :rtype: triple of dataframes
        """
        rows = len(self.df)
        random_df = self.df.iloc[np.random.permutation(rows)].reset_index(drop=True)
        split_index = int(rows * 0.8)
        new_split = split_index + int((rows - split_index) *0.5)
        df_train = random_df[0:split_index]
        df_test = random_df[split_index: new_split].reset_index(drop=True)
        df_validation = random_df[new_split:].reset_index(drop=True)
        return df_train, df_test,df_validation


    def training(self,hp_dim,hp_reg,learning_rate,batch_size,num_steps):
        """ 
        This method creates three batch generators: one for the train df,
        other for the test df, and another for the valid df (this last one will
        creats batchs of the size of the whole valid df); and it also creats
        one object tf_models.SVD (a kind of counterpart of the object SVDmodel 
        that works with tensorflow) and request one training to tf_models.SVD.
        The object tf_models.SVD is save as self.tf_counterpart for the 
        prediction.

        :type hp_dim: int
        :type hp_reg: float
        :type learning_rate: float
        :type batch_size: int
        :type num_steps: int
        """
        self.train_batches = dfFunctions.BatchGenerator(self.train,batch_size,self.users,self.items,self.ratings)
        self.test_batches = dfFunctions.BatchGenerator(self.test,batch_size,self.users,self.items,self.ratings)
        self.valid_batches = dfFunctions.BatchGenerator(self.valid,len(self.valid),self.users,self.items,self.ratings)
        self.tf_counterpart = tf_models.SVD(self.num_of_users,self.num_of_items,self.train_batches,self.test_batches,self.valid_batches)
        self.tf_counterpart.training(hp_dim,hp_reg,learning_rate,num_steps)
        self.tf_counterpart.print_stats()

    def valid_prediction(self):
        """ 
        This method calls the tf_models.SVD and returns the mean
        square error of the whole valid dataset.

        :rtype: float
        """
        return self.tf_counterpart.prediction(show_valid=True)
       
    def prediction(self,list_of_users,list_of_items):
        """ 
        Given one np.array of users and one np.array of items,
        this method calls the tf_models.SVD and returns one np.array
        of predicted ratings.

        :rtype: numpy array of floats
        """
        return self.tf_counterpart.prediction(list_of_users,list_of_items)
        

if __name__ == '__main__':
    import argparse
    path = getcwd() + '/movielens/ml-1m/ratings.dat'

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dimension",type=int, default=15, help="embedding vector size (default=15)")
    parser.add_argument("-r", "--reg",     type=float, default=0.05, help="regularizer constant for the loss function  (default=0.05)")
    parser.add_argument("-l", "--learning", type=float,   default=0.001,   help="learning rate (default=0.001)")
    parser.add_argument("-b", "--batch",type=int, default=1000, help="batch size (default=1000)")
    parser.add_argument("-s", "--steps",type=int, default=5000, help="number of training (default=5000)")
    parser.add_argument("-p", "--path",type=str, default=path, help="ratings path (default=pwd/movielens/ml-1m/ratings.dat)")
    args = parser.parse_args()

    df = dfFunctions.get_data(args.path, sep="::")
    model = SVDmodel(df,'user', 'item','rate')

    dimension = args.dimension
    regularizer_constant = args.reg
    learning_rate = args.learning
    batch_size = args.batch
    num_steps = args.steps

    model.training(dimension,regularizer_constant,learning_rate,batch_size,num_steps)
    prediction = model.valid_prediction()
    print("The mean square error the whole valid dataset is ", prediction) 


        
