
from os import path
import numpy as np
import argparse

import sys
parent_path = path.abspath('..')
sys.path.insert(0, parent_path)
import dfFunctions
import recommender as re

path = parent_path + '/movielens/ml-1m/ratings.dat'

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--path",type=str, default=path, help="ratings path (default=pwd/movielens/ml-1m/ratings.dat)")
parser.add_argument("-e", "--example",type=str, default='1', help="movielens dataset examples (only 1, 10 or 20) (default=1)")
parser.add_argument("-b", "--batch",type=int, default=1000, help="batch size (default=1000)")
parser.add_argument("-s", "--steps",type=int, default=3000, help="number of training steps (default=3000)")
parser.add_argument("-d", "--dimension",type=int, default=15, help="embedding vector size (default=15)")
parser.add_argument("-r", "--reg",     type=float, default=0.05, help="regularizer constant for the loss function  (default=0.05)")
parser.add_argument("-l", "--learning", type=float,   default=0.001,   help="learning rate (default=0.001)")
parser.add_argument("-m", "--momentum",type=float, default=0.9, help="momentum factor (default=0.9)")
args = parser.parse_args()

if args.example == '20':
    path = getcwd() + '/movielens/ml-20m/ratings.csv'
elif args.example == '10':
    path = getcwd() + '/movielens/ml-10m/ratings.dat'
elif args.example =='1':
	pass
else:
    print("Wrong parameter passed to the example option. Running default=1\n")


df = dfFunctions.load_dataframe(args.path)
model = re.NSVDmodel(df,'user', 'item','rating')

dimension = args.dimension
regularizer_constant = args.reg
learning_rate = args.learning
batch_size = args.batch
num_steps = args.steps
momentum_factor = args.momentum

model.training(dimension,regularizer_constant,learning_rate,momentum_factor,batch_size,num_steps)
prediction = model.valid_prediction()
print("\nThe mean square error of the whole valid dataset is ", prediction)
user_example = np.array(model.valid['user'])[0:10]
movies_example = np.array(model.valid['item'])[0:10]
actual_ratings = np.array(model.valid['rating'])[0:10]
predicted_ratings = model.prediction(user_example,movies_example)
print("\nUsing our model for 10 specific users and 10 movies we predicted the following score:")
print(predicted_ratings)
print("\nAnd in reality the scores are:")
print(actual_ratings)