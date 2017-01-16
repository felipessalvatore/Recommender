import tensorflow as tf
import numpy as np
import dfFunctions
from os import getcwd
path = getcwd() + '/movielens/ml-1m/ratings.dat'
df = dfFunctions.load_dataframe(path)
finder = dfFunctions.ItemFinder(df,'user','item','rating')

users = np.array([0,1,2,3,4,5])
items = np.array([0,1,2,3,4,5])
ratings = np.array([2.0,3.0,4.0,2.0,1.0,4.0])

items_per_users = [finder.get_item(i) for i in users]
min_size = min([len(array) for array in items_per_users])
items_per_users = np.array([array[0:min_size] for array in items_per_users])
size_factor = 1/(np.sqrt(min_size))

def inference_nsvd(user_batch,item_batch,user_item_batch,size_factor,user_num, item_num,dim=5):
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

num_of_users= max(df['user']) + 1
num_of_items = max(df['item']) + 1
batch_size = 6
dim = 5
hp_reg = 0.05

graph = tf.Graph()
with graph.as_default():
    tf_user_batch = tf.placeholder(tf.int32, shape=[batch_size], name="id_user")
    tf_item_batch = tf.placeholder(tf.int32, shape=[batch_size], name="id_item")
    tf_user_item = tf.placeholder(tf.int32, shape=[batch_size,None], name="user_item")
    tf_size_factor = tf.placeholder(tf.float32, shape=[], name="size_factor")
    tf_rate_batch = tf.placeholder(tf.float32, shape=[batch_size],name="actual_ratings")
    
    tf_nsvd_model = inference_nsvd(tf_user_batch,tf_item_batch,tf_user_item,tf_size_factor,num_of_users,num_of_items)
    infer, regularizer = tf_nsvd_model['infer'], tf_nsvd_model['regularizer'] 

    with tf.name_scope('loss'):
        tf_cost = loss_function(infer, regularizer,tf_rate_batch,reg=hp_reg)

with tf.Session(graph=graph) as sess: 
    tf.initialize_all_variables().run()
    feed_dict = {tf_user_batch:users,\
                 tf_item_batch:items,\
                 tf_user_item:items_per_users,\
                 tf_size_factor: size_factor,\
                tf_rate_batch:ratings}
    a,b,c = sess.run([infer,regularizer,tf_cost],feed_dict=feed_dict)
    print(a)
    print(b)
    print(c)