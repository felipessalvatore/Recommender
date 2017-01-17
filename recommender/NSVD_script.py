if __name__ == '__main__':

	import numpy as np
	import dfFunctions
	import tf_models
	import recommender as re
	from os import getcwd
	path = getcwd() + '/movielens/ml-1m/ratings.dat'
	df = dfFunctions.load_dataframe(path)
	model = re.NSVDmodel(df,'user', 'item','rating')

	dimension = 10
	regularizer_constant = 0.05
	learning_rate = 0.0001
	batch_size = 1000
	num_steps = 9000
	momentum_factor = 0.001

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
