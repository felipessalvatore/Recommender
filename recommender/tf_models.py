import tensorflow as tf
import numpy as np
from utils import rmse,status_printer
import time
import os

def inference_svd(user_batch, item_batch, user_num, item_num, dim=5):
    """
    This function creates one tensor of shape=[dim] for every user
    and every item. We select the indices for users from the tensor user_batch
    and select the indices for items from the tensor item_batch. After that we
    calculate the infered score as the inner product between the user vector and
    the item vector (we also sum the global bias, the bias from that user and 
    the bias from that item). infer is the tensor with the result of this
    caculation.

    We calculate also a regularizer to use in the loss function. This function
    returns a dictionary with the tensors infer, regularizer, w_user (tensor with
    all the user vectors) and w_items (tensor with all the item vectors).

    :type item_batch: tensor of int32
    :type user_batch: tensor of int32
    :type user_num: int
    :type item_num: int
    :type dim: int
    :rtype: dictionary

    """
    with tf.name_scope('Declaring_variables'):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.name_scope('Prediction_regularizer'):
        infer = tf.reduce_sum(tf.mul(embd_user, embd_item), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        l2_user = tf.sqrt(tf.nn.l2_loss(embd_user))
        l2_item = tf.sqrt(tf.nn.l2_loss(embd_item))
        bias_user_sq = tf.square(bias_user)
        bias_item_sq = tf.square(bias_item)
        bias_sum = tf.add(bias_user_sq,bias_item_sq)
        l2_sum = tf.add(l2_user, l2_item)
        regularizer = tf.add(l2_sum, bias_sum, name="svd_regularizer")
    dic_of_values = {'infer': infer, 'regularizer': regularizer, 'w_user': w_user, 'w_item': w_item}    
    return dic_of_values


def inference_nsvd(user_batch,item_batch,user_item_batch,size_factor,user_num, item_num,dim=5):
    """
    Escrever
    :type item_batch: tensor of int32
    :type user_batch: tensor of int32
    :type user_num: int
    :type item_num: int
    :type dim: int
    :rtype: dictionary
    """
    with tf.name_scope('Declaring_variables'):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_item1 = tf.get_variable(name='w_item1',shape=[item_num,dim],
                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item2 = tf.get_variable(name='w_item2',shape=[item_num,dim],
                         initializer=tf.truncated_normal_initializer(stddev=0.02))
        embd_item1 = tf.nn.embedding_lookup(w_item1,item_batch)
        embd_item2 = tf.nn.embedding_lookup(w_item2,user_item_batch)
        embd_item2 = tf.mul(tf.reduce_sum(embd_item2,1),size_factor)
    with tf.name_scope('Prediction_regularizer'):
        infer = tf.reduce_sum(tf.mul(embd_item1,embd_item2), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        l2_user = tf.sqrt(tf.nn.l2_loss(embd_item1))
        l2_item = tf.sqrt(tf.nn.l2_loss(embd_item2))
        bias_user_sq = tf.square(bias_user)
        bias_item_sq = tf.square(bias_item)
        bias_sum = tf.add(bias_user_sq,bias_item_sq)
        l2_sum = tf.add(l2_user, l2_item)
        regularizer = tf.add(l2_sum, bias_sum, name="svd_regularizer")
    dic_of_values = {'infer': infer, 'regularizer': regularizer, 'w_item1': w_item1, 'w_item2': w_item2}
    return dic_of_values


def loss_function(infer, regularizer, rate_batch,reg):
    """
    Given one tensor with all the predictions from the batch (infer)
    and one tensor with all the real scores from the batch (rate_batch)
    we calculate, using numpy sintax, cost_l2 = np.sum((infer - rate_batch)**2)
    After that this function return cost_l2 + lambda3*regularizer.

    :type infer: tensor of float32
    :type regularizer: tensor, shape=[],dtype=float32
    :type rate_batch: tensor of int32
    :type reg: float
    """
    cost_l2 = tf.square(tf.sub(rate_batch,infer))
    lambda3 = tf.constant(reg, dtype=tf.float32, shape=[], name="lambda3")
    cost = tf.add(cost_l2, tf.mul(regularizer, lambda3))
    return cost


class SVD(object):
    """
    Class specialized in communicating with tensorflow. It receives all
    data information from the class recommender.SVDmodel and sets the
    tensorflow graph, it also run the graph in a Session for training
    and for prediction.

    :type num_of_users: int
    :type num_of_items: int
    :type train_batch_generator: dfFunctions.BatchGenerator
    :type test_batch_generator: dfFunctions.BatchGenerator
    :type valid_batch_generator: dfFunctions.BatchGenerator
    """
    def __init__(self,num_of_users,num_of_items,train_batch_generator,test_batch_generator,valid_batch_generator):
        self.num_of_users = num_of_users
        self.num_of_items = num_of_items
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.general_duration = 0
        self.num_steps = 0
        self.dimension = None
        self.regularizer = None
        self.best_acc_test = float('inf')


    def set_graph(self,hp_dim,hp_reg,learning_rate,momentum_factor):
        """
        This function only sets the tensorflow graph and stores it
        as self.graph. Here we do not keep the log to pass it to
        Tensorboard. We save the params hp_dim, hp_reg and learning_rate
        as self.dimension, self.regularizer, self.learning_rate,
        respectively, in order to get the same graph while doing the
        prediction.

        :type hp_dim: int
        :type hp_reg: float
        :type learning_rate: float
        :type momentum_factor: float
        """
        self.dimension = hp_dim
        self.regularizer = hp_reg
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.graph = tf.Graph()
        with self.graph.as_default():

            #Placeholders
            self.tf_user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
            self.tf_item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
            self.tf_rate_batch = tf.placeholder(tf.float32, shape=[None],name="actual_ratings")

            #Applying the model
            tf_svd_model = inference_svd(self.tf_user_batch, self.tf_item_batch, user_num=self.num_of_users, item_num=self.num_of_items, dim=hp_dim)
            self.infer, regularizer = tf_svd_model['infer'], tf_svd_model['regularizer'] 

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                self.tf_cost = loss_function(self.infer, regularizer,self.tf_rate_batch,reg=hp_reg)

            #Optimizer
            with tf.name_scope('training'):
                global_step = tf.contrib.framework.assert_or_get_global_step()
                assert global_step is not None
                self.train_op = tf.train.MomentumOptimizer(learning_rate,momentum_factor).minimize(self.tf_cost, global_step=global_step)

            #Saver
            self.saver = tf.train.Saver()
            save_dir = 'checkpoints/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_path = os.path.join(save_dir, 'best_validation')

            #Minibatch accuracy using rmse
            with tf.name_scope('accuracy'):
                self.acc_op =  tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(self.infer,self.tf_rate_batch),2)))


    def training(self,hp_dim,hp_reg,learning_rate,momentum_factor,num_steps):
        """
        After created the graph this function run it in a Session for
        training. We print some information just to keep track of the
        training. Every time the accuracy of the test batch is decrease
        we save the variables of the model (we use * to mark a new save).


        :type hp_dim: int
        :type hp_reg: float
        :type learning_rate: float
        :type momentum_factor: float
        :type num_steps: int
        """
        self.set_graph(hp_dim,hp_reg,learning_rate,momentum_factor)
        self.num_steps = num_steps
        marker = ''

        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            print("{} {} {} {}".format("step", "batch_error", "test_error","elapsed_time"))
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = self.train_batch_generator.get_batch()
                feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}         
                _, pred_batch,cost,train_error = sess.run([self.train_op, self.infer, self.tf_cost,self.acc_op], feed_dict=feed_dict)
                if (step % 1000)  == 0:
                    users, items, rates = self.test_batch_generator.get_batch() 
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}              
                    pred_batch = sess.run(self.infer, feed_dict=feed_dict)
                    test_error = rmse(pred_batch,rates)
                    if test_error < self.best_acc_test:
                        self.best_acc_test = test_error
                        marker = "*"
                        self.saver.save(sess=sess, save_path=self.save_path)

                    end = time.time()
                    print("{:3d} {:f} {:f}{:s} {:f}(s)".format(step,train_error,test_error,marker,
                                                           end - start))
                    marker = ''
                    start = end
        self.general_duration = time.time() - initial_time

    def print_stats(self):
        """
        Method that calls the status_printer function.
        This method can be called before the training, but it will only print
        that the training lasted 0 seconds.
        """
        status_printer(self.num_steps,self.general_duration)

    def prediction(self,list_of_users=None,list_of_items=None,show_valid=False):
        """
        Prediction function. This function loads the tensorflow graph
        with the same params from the training and with the saved
        variables. The user can either check what is the mean square error
        of the whole valid dataset (if show_valid == True),  or the user
        can use two np.arrays of the same size (one is a list of users
        and the other is a list of items) and this function will return
        what is the predicted score (as a np array of floats).

        :type list_of_users: numpy array of ints
        :type list_of_items: numpy array of ints
        :type show_valid: boolean
        :rtype valid_error: float
        :rtype predicion: numpy array of floats
        """
        if self.dimension == None and self.regularizer == None:
            print("You can not have a prediction without training!!!!")
        else:
            self.set_graph(self.dimension,self.regularizer,self.learning_rate,self.momentum_factor)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                users, items, rates = self.valid_batch_generator.get_batch()
                if show_valid:
                    feed_dict = {self.tf_user_batch: users, self.tf_item_batch: items, self.tf_rate_batch: rates}
                    valid_error = sess.run(self.acc_op, feed_dict=feed_dict)
                    return valid_error
                else:
                    feed_dict = {self.tf_user_batch: list_of_users, self.tf_item_batch: list_of_items}
                    prediction = sess.run(self.infer, feed_dict=feed_dict)
                    return prediction



class NSVD(object):
    """
    Class specialized in communicating with tensorflow. It receives all
    data information from the class recommender.NSVDmodel and sets the
    tensorflow graph, it also run the graph in a Session for training
    and for prediction.

    :type num_of_users: int
    :type num_of_items: int
    :type train_batch_generator: dfFunctions.BatchGenerator
    :type test_batch_generator: dfFunctions.BatchGenerator
    :type valid_batch_generator: dfFunctions.BatchGenerator
    :type finder: dfFunctions.ItemFinder
    """
    def __init__(self,num_of_users,num_of_items,train_batch_generator,test_batch_generator,valid_batch_generator,finder):
        self.num_of_users = num_of_users
        self.num_of_items = num_of_items
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.finder = finder
        self.general_duration = 0
        self.num_steps = 0
        self.dimension = None
        self.regularizer = None
        self.best_acc_test = float('inf')


    def set_graph(self,hp_dim,hp_reg,learning_rate,momentum_factor):
        """
        This function only sets the tensorflow graph and stores it
        as self.graph. Here we do not keep the log to pass it to
        Tensorboard. We save the params hp_dim, hp_reg and learning_rate
        as self.dimension, self.regularizer, self.learning_rate,
        respectively, in order to get the same graph while doing the
        prediction.

        :type hp_dim: int
        :type hp_reg: float
        :type learning_rate: float
        :type momentum_factor: float
        """
        self.dimension = hp_dim
        self.regularizer = hp_reg
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.graph = tf.Graph()
        with self.graph.as_default():

            #Placeholders
            self.tf_user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
            self.tf_item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
            self.tf_rate_batch = tf.placeholder(tf.float32, shape=[None],name="actual_ratings")
            self.tf_user_item = tf.placeholder(tf.int32, shape=[None,None], name="user_item")
            self.tf_size_factor = tf.placeholder(tf.float32, shape=[], name="size_factor")

            #Applying the model
            tf_nsvd_model = inference_nsvd(self.tf_user_batch, self.tf_item_batch,self.tf_user_item,self.tf_size_factor, user_num=self.num_of_users, item_num=self.num_of_items, dim=hp_dim)
            self.infer, regularizer = tf_nsvd_model['infer'], tf_nsvd_model['regularizer'] 

            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                self.tf_cost = loss_function(self.infer, regularizer,self.tf_rate_batch,reg=hp_reg)

            #Optimizer
            with tf.name_scope('training'):
                global_step = tf.contrib.framework.assert_or_get_global_step()
                assert global_step is not None
                self.train_op = tf.train.MomentumOptimizer(learning_rate,momentum_factor).minimize(self.tf_cost, global_step=global_step)

            #Saver
            self.saver = tf.train.Saver()
            save_dir = 'checkpoints/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_path = os.path.join(save_dir, 'best_validation')

            #Minibatch accuracy using rmse
            with tf.name_scope('accuracy'):
                self.acc_op =  tf.sqrt(tf.reduce_mean(tf.pow(tf.sub(self.infer,self.tf_rate_batch),2)))


    def training(self,hp_dim,hp_reg,learning_rate,momentum_factor,num_steps):
        """
        After created the graph this function run it in a Session for
        training. We print some information just to keep track of the
        training. Every time the accuracy of the test batch is decrease
        we save the variables of the model (we use * to mark a new save).


        :type hp_dim: int
        :type hp_reg: float
        :type learning_rate: float
        :type momentum_factor: float
        :type num_steps: int
        """
        self.set_graph(hp_dim,hp_reg,learning_rate,momentum_factor)
        self.num_steps = num_steps
        marker = ''

        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            print("{} {} {} {}".format("step", "batch_error", "test_error","elapsed_time"))
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = self.train_batch_generator.get_batch()
                items_per_user =self.finder.get_item_array(users)
                size_factor = self.finder.size_factor
                feed_dict = {self.tf_user_batch: users,\
                             self.tf_item_batch: items,\
                             self.tf_rate_batch: rates,\
                             self.tf_user_item:items_per_user,\
                             self.tf_size_factor:size_factor}         
                _, pred_batch,cost,train_error = sess.run([self.train_op, self.infer, self.tf_cost,self.acc_op], feed_dict=feed_dict)
                if (step % 1000)  == 0:
                    users, items, rates = self.test_batch_generator.get_batch() 
                    items_per_user = self.finder.get_item_array(users)
                    size_factor = self.finder.size_factor
                    feed_dict = {self.tf_user_batch: users,\
                             self.tf_item_batch: items,\
                             self.tf_rate_batch: rates,\
                             self.tf_user_item:items_per_user,\
                             self.tf_size_factor:size_factor}     
                    pred_batch = sess.run(self.infer, feed_dict=feed_dict)
                    test_error = rmse(pred_batch,rates)
                    if test_error < self.best_acc_test:
                        self.best_acc_test = test_error
                        marker = "*"
                        self.saver.save(sess=sess, save_path=self.save_path)

                    end = time.time()
                    print("{:3d} {:f} {:f}{:s} {:f}(s)".format(step,train_error,test_error,marker,
                                                           end - start))
                    marker = ''
                    start = end
        self.general_duration = time.time() - initial_time

    def print_stats(self):
        """
        Method that calls the status_printer function.
        This method can be called before the training, but it will only print
        that the training lasted 0 seconds.
        """
        status_printer(self.num_steps,self.general_duration)

    def prediction(self,list_of_users=None,list_of_items=None,show_valid=False):
        """
        Prediction function. This function loads the tensorflow graph
        with the same params from the training and with the saved
        variables. The user can either check what is the mean square error
        of the whole valid dataset (if show_valid == True),  or the user
        can use two np.arrays of the same size (one is a list of users
        and the other is a list of items) and this function will return
        what is the predicted score (as a np array of floats).

        :type list_of_users: numpy array of ints
        :type list_of_items: numpy array of ints
        :type show_valid: boolean
        :rtype valid_error: float
        :rtype predicion: numpy array of floats
        """
        if self.dimension == None and self.regularizer == None:
            print("You can not have a prediction without training!!!!")
        else:
            self.set_graph(self.dimension,self.regularizer,self.learning_rate,self.momentum_factor)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                users, items, rates = self.valid_batch_generator.get_batch()
                if show_valid:
                    items_per_user = self.finder.get_item_array(users)
                    size_factor = self.finder.size_factor
                    feed_dict = {self.tf_user_batch: users,\
                             self.tf_item_batch: items,\
                             self.tf_rate_batch: rates,\
                             self.tf_user_item:items_per_user,\
                             self.tf_size_factor:size_factor}   
                    valid_error = sess.run(self.acc_op, feed_dict=feed_dict)
                    return valid_error
                else:
                    items_per_user = self.finder.get_item_array(list_of_users)
                    size_factor = self.finder.size_factor
                    feed_dict = {self.tf_user_batch: list_of_users,\
                                 self.tf_item_batch: list_of_items,\
                                 self.tf_user_item:items_per_user,\
                                 self.tf_size_factor:size_factor}  
                    prediction = sess.run(self.infer, feed_dict=feed_dict)
                    return prediction
