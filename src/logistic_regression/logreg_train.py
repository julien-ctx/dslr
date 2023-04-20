import sys, os
import pandas as pd
import numpy as np

# https://fr.wikipedia.org/wiki/Encodage_one-hot
# https://en.wikipedia.org/wiki/Softmax_function
# https://en.wikipedia.org/wiki/Sigmoid_function

class LogisticRegression:
	def __init__(self, df):
		self.df  = df.drop('Index', axis=1).drop(df.columns[2:6], axis=1)
 		
	# All necessary operations to compute the prediction without any error.
	def preprocess_data(self):
		self.convert()
		self.interpolate()
		self.standardize()
		self.sample_size = self.df.shape[0]
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to 0.0 for the moment.
		self.weights = np.random.randn(self.df.shape[1], 4) * 0.01
		# Bias is added to take into account every value independently from their value.
		self.bias = np.ones(4)
		self.logits = self.df.to_numpy() @ self.weights + self.bias

	def convert(self):
		self.df['Hogwarts House'] = self.df['Hogwarts House'].replace('Hufflepuff', 1).replace('Gryffindor', 2).replace('Ravenclaw', 3).replace('Slytherin', 4)
 
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

	def get_weights_gradient(self, y_binary):
		return (self.df.to_numpy().T @ (self.sigmoid() - y_binary)) / self.df.shape[0]

	def gradient_descent(self, y_binary):
		alpha = 0.001
		for _ in range(10000):
			self.weights = self.weights - alpha * self.get_weights_gradient(y_binary)
		# self.logits = self.df.to_numpy() @ self.weights + self.bias
		print(self.sigmoid()[1])

	# Get probability with sigmoid function
	def sigmoid(self):
		return 1 / (1 + np.exp(-(np.array(self.df @ self.weights))))
	
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
	houses = ['Hufflepuff', 'Gryffindor', 'Ravenclaw', 'Slytherin']
	for house in houses:
		y_binary = np.array((df['Hogwarts House'] == house).astype(float))
		y_binary = np.reshape(y_binary, (1600, 1))
		model.sigmoid()
		model.gradient_descent(y_binary)
