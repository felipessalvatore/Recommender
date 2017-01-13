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

def get_data(filname, sep="\t"):
    """
    Given one filepath filename and one separator sep, 
    it returns one dataframe with columns user (int),
    item (int) and rate (float).


    :type filename: string  
    :type sep: string
    :rtype: dataframe
    """
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


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
