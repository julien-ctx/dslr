import sys, os
import pandas as pd
import numpy as np

# https://fr.wikipedia.org/wiki/Encodage_one-hot
# https://en.wikipedia.org/wiki/Softmax_function
# https://en.wikipedia.org/wiki/Sigmoid_function

class LogisticRegression:
	def __init__(self, df):
		self.df = df.iloc[:, 6:]
 		
	# All necessary operations to compute the prediction without any error.
	def preprocess_data(self):
		self.interpolate()
		self.standardize()
		self.sample_size = self.df.shape[0]
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to 0.0 for the moment.
		self.weights = np.zeros((self.df.shape[1], 4), dtype=float)
		# Bias is added to take into account every value independently from their value.
		self.bias = np.ones(4)
		self.logits = self.df.to_numpy() @ self.weights + self.bias

	# We fill NaN and null values with the mean of the column.
	# It is necessary to avoid decreasing data accuracy.
	def interpolate(self):
		for x in range(self.df.shape[1]):
			for y in range(self.df.shape[0]):
				if pd.isna(self.df.iloc[y, x]):
					self.df.iloc[y, x] = self.df.iloc[:, x].mean()

	# Standardize data to improve performance and big number issues.
	def standardize(self):
		self.df = self.df.apply(lambda x : (x - np.mean(x)) / np.std(x))

	def get_one_hot(self):
		houses = ['Ravenclaw', 'Slytherin', 'Hufflepuff', 'Gryffindor']
		houses_one_hot = []
		for house in houses:
			house_one_hot = [0] * len(houses)
			house_one_hot[houses.index(house)] = 1
			houses_one_hot.append(house_one_hot)
		houses_one_hot = np.tile(houses_one_hot, (self.sample_size, 1))
		houses_one_hot = houses_one_hot[:self.sample_size, :]
		return houses_one_hot

	def get_weights_gradient(self, one_hot):
		a = self.proba - one_hot
		return (self.df.to_numpy().T @ (self.proba - one_hot)) / self.df.shape[0]

	def gradient_descent(self):
		alpha = 0.001
		one_hot = self.get_one_hot()
		for i in range(10000):
			self.weights = self.weights - alpha * self.get_weights_gradient(one_hot)
			self.bias = sum(self.proba - one_hot) / self.sample_size
		self.logits = self.df.to_numpy() @ self.weights + self.bias
		self.activation()
		print(self.proba)

	# Get probability with sigmoid function
	def activation(self, z):
		self.proba = 1 / (1 + np.exp(-z))
		print(self.proba)
		exit()
		# exp = np.exp(self.logits)
		# self.proba = exp / np.sum(exp, axis=1, keepdims=True)
	
	# Data init
	def init_data(self):
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to 0.0 for the moment.
		self.weights = np.zeros((self.df.shape[1], 4), dtype=float)
		# Bias is added to take into account every value independently from their value.
		self.bias = np.ones(4)
		self.logits = self.df.to_numpy() @ self.weights + self.bias
 
if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	model = LogisticRegression(df)

	model.preprocess_data()
	model.activation(model.df.to_numpy() @ model.weights)
	model.gradient_descent()
