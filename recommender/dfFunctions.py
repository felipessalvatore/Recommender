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
    In implementing the NSVD model I run in the following problem.
    The user factor vector should be represented by
    np.sum(R(u),1)*(1/np.sqrt(len(R(u)))) where R(u) is the array of all items
    rated by u. But it turns out that I found major difficulties in creating
    tensors in tensorflow that have as elements arrays with different sizes.

    In this dataset the number of rated items per user is very different: some
    is around 1000 others around 20. And in each minibatch of users I could not
    only pass theses arrays with their raw shapes (since they
    have different sizes).So I decided to normalize all the arrays
    of rated items. The method set_item_dic, creates a dictionary of
    users and rated items, find the smallest size of an array of rated items,
    say n; and after that slice all the arrays in order to have them with size
    n. Hence every time when the class tf_models.SVD selects a batch of m
    users it feeds to the tensorflow graph a matrix of shape=[m,n].

    Another option is fill each array of set_item_dic with a number of a fake
    item like new_item = max(self.df[self.items].unique()) +1 creat a vector
    in tensorflor with zeros only and them concat it with the items vector.
    Similar as in the following script:

    import tensorflow as tf
    import numpy as np

    ids = np.array([1,2,10])
    zero = tf.constant(np.array([[0,0]]),dtype="float32")
    initializer = tf.truncated_normal_initializer(mean=3,stddev=0.02)
    w_item = tf.get_variable("embd_user", shape=[10, 2],
                             initializer=initializer)
    w_item = tf.concat(0,[w_item, zero])
    result = tf.nn.embedding_lookup(w_item, ids)
    with tf.Session() as sess:
    tf.initialize_all_variables().run()
    a = sess.run(result)
    print(a)

    example for multiply each vector by an scalar:

    import tensorflow as tf
    import numpy as np


    a1 = tf.constant(np.array([[1,10],[2,3]]),shape=[2,2],dtype="float32")
    a2 = tf.constant(np.array([-1,2]),shape=[2],dtype="float32")
    a1 = tf.transpose(a1)
    result = tf.mul(a1, a2)
    a1 = tf.transpose(result)

    with tf.Session() as sess:
        print(sess.run(a1))

    Felipe (19/01/17).


    :type df: dataframe
    :type users: string
    :type items: string
    :type ratings: string
    """

    def __init__(self, df, users, items, ratings):
        self.users = users
        self.items = items
        self.df = df
        self.dic = {}
        self._set_item_dic()

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

    def _set_item_dic(self):
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
            self.max_size = max(sizes.values())
            print("Resizing ...")
            for user in all_users:
                difference_of_sizes = self.max_size - sizes[user]
                if difference_of_sizes > 0:
                    tail = [new_item for i in range(difference_of_sizes)]
                    tail = np.array(tail)
                    result = np.concatenate((self.dic[user], tail), axis=0)
                    result = result.astype(np.int32)
                    self.dic[user] = result
            print("Generating size factors ...")
            for user in all_users:
                sizes[user] = 1/np.sqrt(sizes[user])
            self.size_factor = sizes
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
