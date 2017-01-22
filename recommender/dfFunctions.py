import numpy as np
import pandas as pd


def load_dataframe(path, sep="::"):
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
        col_names = ["userId", "movieId", "rating", "st"]
        raw_df = pd.read_csv(path, sep=sep, names=col_names, engine='python')
    elif path[-3:] == 'csv':
        raw_df = pd.read_csv(path)
    raw_df['userId'] = raw_df['userId'] - 1
    raw_df['movieId'] = raw_df['movieId'] - 1
    raw_df['user'] = raw_df["userId"].astype(np.int32)
    raw_df['item'] = raw_df["movieId"].astype(np.int32)
    raw_df["rating"] = raw_df["rating"].astype(np.float32)
    df = raw_df[["user", "item", "rating"]]
    return df


def count_intersection(df1, df2, df3):
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
    set1 = set([sha1(observation).hexdigest()
                for observation in array1])
    set2 = set([sha1(observation).hexdigest()
               for observation in array2])
    set3 = set([sha1(observation).hexdigest()
               for observation in array3])
    dic = {}
    dic['1-2'] = len(set1.intersection(set2))
    dic['1-3'] = len(set1.intersection(set3))
    dic['2-3'] = len(set2.intersection(set3))
    return dic


class ItemFinder(object):
    """
    Class that given one user it returns
    the array of all items rated by that user.


    :type df: dataframe
    :type users: string
    :type items: string
    :type ratings: string
    """

    def __init__(self, df, users, items, ratings, nsvd_size):
        self.users = users
        self.items = items
        self.df = df
        self.dic = {}
        self._set_item_dic(nsvd_size)

    def get_item(self, user):
        """
        Every time we call this method it returns
        the array of items rated by the user

        :type user: int
        :rtype: numpy array
        """
        user_df = self.df[self.df[self.users] == user]
        user_items = np.array(user_df[self.items])
        return user_items

    def _set_item_dic(self, size_command="mean"):
        """
        This method returns a dic: user:array_of_rated_items.
        The size of array_of_rated_items is the size of
        the smallest number of rated items from an user.

        :rtype items_per_users: dic
        """
        if not self.dic:
            all_users = self.df[self.users].unique()
            new_item = max(self.df[self.items].unique()) + 1
            sizes = {}
            print("\nWriting dic ...")
            for user in all_users:
                items_rated = self.get_item(user)
                self.dic[user] = items_rated
                sizes[user] = len(items_rated)
            if size_command == "max":
                self.size = np.max(list(sizes.values()))
            elif size_command == "mean":
                self.size = int(np.mean(list(sizes.values())))
            elif size_command == "min":
                self.size = np.min(list(sizes.values()))
            print("Resizing ...")
            for user in all_users:
                if self.size <= sizes[user]:
                    self.dic[user] = self.dic[user][0:self.size]
                else:
                    difference_of_sizes = self.size - sizes[user]
                    tail = [new_item for i in range(difference_of_sizes)]
                    tail = np.array(tail)
                    result = np.concatenate((self.dic[user], tail), axis=0)
                    result = result.astype(np.int32)
                    self.dic[user] = result
            print("Generating size factors ...")
            if size_command == "max":
                for user in all_users:
                    sizes[user] = 1/np.sqrt(sizes[user])
                self.size_factor = sizes
            else:
                self.size_factor = dict.fromkeys(sizes, 1/np.sqrt(self.size))
        else:
            pass

    def get_item_array(self, users):
        """
        Given the list user =[u1, ..., un]
        this method returns the array [r1, ..., rn]
        where ri is the array_of_rated_items by the user
        ui according the dictionary self.dic.

        :type users: numpy array,dtype=int
        :rtype: numpy array,dtype=int
        """

        return np.array([self.dic[user] for user in users])

    def get_size_factors(self, users):
        """
        Given the list user =[u1, ..., un]
        this method returns the array [f1, ..., fn]
        where fi is the size factor of user
        ui according the dictionary self.size_factor.

        :type users: numpy array,dtype=int
        :rtype: numpy array,dtype=float
        """

        return np.array([self.size_factor[user] for user in users])


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

    def __init__(self, df, batch_size, users, items, ratings):
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
        random_indices = np.random.randint(0, self.size, self.batch_size)
        users = self.users[random_indices]
        items = self.items[random_indices]
        ratings = self.ratings[random_indices]
        return users, items, ratings
