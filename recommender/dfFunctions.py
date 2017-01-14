import numpy as np
import pandas as pd

def load_dataframe(path,sep="::"):
    """
    Given one filepath path and one separator sep,
    it returns one dataframe with columns user (int),
    item (int) and ratings (float). This function assumes
    that we are working only the datasets from movielens.


    :type path: string
    :type sep: string
    :rtype: dataframe
    """
    if path[-3:] == 'dat':
        col_names = ["raw_user", "raw_item", "raw_rating", "st"]
        raw_df = pd.read_csv(path, sep=sep,names=col_names,engine='python')
        raw_df['user'] = raw_df["raw_user"].astype(np.int32)
        raw_df['item'] = raw_df["raw_item"].astype(np.int32)
        raw_df["rating"] = raw_df["raw_rating"].astype(np.float32)
        df = raw_df[["user", "item", "rating"]]
        return df
    elif path[-3:] == 'csv':
        raw_df = pd.read_csv(path)
        raw_df['user'] = raw_df["userId"].astype(np.int32)
        raw_df['item'] = raw_df["movieId"].astype(np.int32)
        raw_df["rating"] = raw_df["rating"].astype(np.float32)
        df = raw_df[["user", "item", "rating"]]
        return df


def count_intersection(df1,df2,df3):
    """
    Given three dataframes df1,df2 and df3, this function
    counts how many shared observations these dataframes have.
    We work with three dataframes because the intended use for 
    this function is to deal with the train, test and valid 
    dataframes.

    This function returns a dictionary with the keys '1-2',
    '1-3' and '2-3' representing the intersection between
    df1 and df2, the intersection between df1 and df3 and
    the intersection between df2 and df3, respectively.


    :type df1: datatframe
    :type df2: dataframe
    :type df3: dataframe
    :rtype: dictionary
    """
    from hashlib import sha1
    raw_array1 = np.array(df1)
    raw_array2 = np.array(df2)
    raw_array3 = np.array(df3)
    array1 = raw_array1.copy(order='C')
    array2 = raw_array2.copy(order='C')
    array3 = raw_array3.copy(order='C')
    set1 = set([sha1(observation).hexdigest()\
     for observation in array1])
    set2 = set([sha1(observation).hexdigest()\
     for observation in array2])
    set3 = set([sha1(observation).hexdigest()\
     for observation in array3])
    dic = {}
    dic['1-2'] = len(set1.intersection(set2))
    dic['1-3'] = len(set1.intersection(set3))
    dic['2-3'] = len(set2.intersection(set3))
    return dic 







class BatchGenerator(object):
    """
    Class to generate batches using one dataframe and one number
    of batch size. The params users, items and ratings are the names
    from the columns of this dataset that have the user, the items and
    the score information, respectively.


    :type df: dataframe
    :type batch_size: int
    :type users: string
    :type items: string
    :type ratings: string
    """

    def __init__(self,df,batch_size,users,items,ratings):
        self.batch_size = batch_size
        self.users = np.array(df[users])
        self.items = np.array(df[items])
        self.ratings = np.array(df[ratings])
        self.num_cols = len(df.columns)
        self.size = len(df)

    def get_batch(self):
        """
        Every time we call this method, a new list of size batch_size
        with random numbers is created (all numbers less than the size
        of the dataframe). With this list we select some random users,
        items and ratings.

        :rtype: triple of numpy arrays
        """
        random_indices = np.random.randint(0,self.size,self.batch_size)
        users = self.users[random_indices]
        items = self.items[random_indices]
        ratings = self.ratings[random_indices]
        return users, items, ratings
