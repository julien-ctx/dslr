import sys, os
import pandas as pd
import numpy as np

# https://fr.wikipedia.org/wiki/Encodage_one-hot
# https://en.wikipedia.org/wiki/Softmax_function

class LogisticRegression:
	def __init__(self, df):
		self.df = df
		self.clean_data()
		self.standardize()

	def get_one_hot(self):
		houses = ['Ravenclaw', 'Slytherin', 'Hufflepuff', 'Gryffindor']
		houses_one_hot = []
		for house in houses:
			house_one_hot = [0] * len(houses)
			house_one_hot[houses.index(house)] = 1
			houses_one_hot.append(house_one_hot)
		houses_one_hot = np.tile(houses_one_hot, (1251, 1))
		houses_one_hot = houses_one_hot[:1251, :]
		return houses_one_hot

	def get_weights_gradient(self, one_hot):
		a = self.proba - one_hot
		return self.df.to_numpy().T @ (self.proba - one_hot) / self.df.shape[0]

	def gradient_descent(self):
		alpha = 0.001
		one_hot = self.get_one_hot()
		for i in range(100000):
			self.weights = self.weights - alpha * self.get_weights_gradient(one_hot)
			self.bias = sum(self.proba - one_hot) / 1251
		self.logits = self.df.to_numpy() @ self.weights + self.bias
		self.softmax()
		print(self.proba)

	# Get probability
	def softmax(self):
		exp = np.exp(self.logits)
		self.proba = exp / np.sum(exp, axis=1, keepdims=True)
	
	# Data init
	def init_data(self):
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to 0.0 for the moment.
		self.weights = np.zeros((self.df.shape[1], 4), dtype=float)
		# Bias is added to take into account every value independently from their value.
		self.bias = np.ones(4)
		self.logits = self.df.to_numpy() @ self.weights + self.bias
 
	# Data preprocessing
	def clean_data(self):
		new_df = self.df.dropna()
		self.df = new_df.iloc[:, 6:]

	def standardize(self):
		self.df = self.df.apply(lambda x : (x - np.mean(x)) / np.std(x))
 
	def get_df(self):
		return self.df

if __name__ == "__main__":
	# Input list of cities
	cities = ["Paris", "Bangkok", "London", "Atlanta"]
	onehot_encoded = []

	# Iterate through each city in the original list
	

	# Convert the result array to a NumPy array
	onehot_encoded = np.array(onehot_encoded)

	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	model = LogisticRegression(df)

	model.init_data()
	model.softmax()
	model.gradient_descent()
