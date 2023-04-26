import sys, os
import pandas as pd
import numpy as np

# https://fr.wikipedia.org/wiki/Encodage_one-hot
# https://en.wikipedia.org/wiki/Softmax_function
# https://en.wikipedia.org/wiki/Sigmoid_function

class LogisticRegressionTrain:
	def __init__(self, df):
		self.df = df
		self.df = self.df.drop('Index', axis=1).drop(self.df.columns[1:6], axis=1)

	# All necessary operations to compute the prediction without any error.
	def preprocess_data(self):
		self.interpolate()
		self.standardize()
		# Bias is added to take into account every value independently from their value.
		self.df['Bias'] = np.ones(self.df.shape[0])
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to 0.0 for the moment.
		self.weights = np.random.randn(self.df.shape[1], 1) * 0.01

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

	def get_weights(self, y_binary):
		return (self.df.to_numpy().T @ (self.sigmoid() - y_binary)) / self.df.shape[0]

	def gradient_descent(self, y_binary, house, weights_df):
		base_filename = 'eval_'
		n_files = len([f for f in os.listdir('../../assets') if os.path.isfile(f) and f.startswith(base_filename)])
		eval_file = open(f'../../assets/{base_filename}{n_files}.csv', 'w+')

		alpha = 0.001
		for _ in range(10000):
			self.weights = self.weights - alpha * self.get_weights(y_binary)
			if _ % 100 == 0:
				probabilities = self.sigmoid()
				loss = -np.mean(y_binary * np.log(probabilities) + (1 - y_binary) * np.log(1 - probabilities))
				# Print or store the loss, for example, appending it to a list
				eval_file.write(f'{loss}\n')

		# self.logits = self.df.to_numpy() @ self.weights + self.bias
		return pd.concat([weights_df, pd.DataFrame(self.weights, columns=[house])], axis=1)

	# Get probability with sigmoid function
	def sigmoid(self):
		return 1 / (1 + np.exp(-(np.array(self.df @ self.weights))))


class LogisticRegressionPredict:
	def __init__(self, sample_path, weights_path):	
		self.sample_df = pd.read_csv(sample_path)
		self.sample_df = self.sample_df.drop('Index', axis=1).drop(self.sample_df.columns[1:6], axis=1)
		self.sample_df = self.interpolate()
		self.standardize()
		self.sample_df['Bias'] = np.ones(self.sample_df.shape[0])
		self.weights_df = pd.read_csv(weights_path)	

	def interpolate(self):
		for x in range(self.sample_df.shape[1]):
			for y in range(self.sample_df.shape[0]):
				if pd.isna(self.sample_df.iloc[y, x]):
					self.sample_df.iloc[y, x] = self.sample_df.iloc[:, x].mean()
		return self.sample_df
	
	def standardize(self):
		self.sample_df = self.sample_df.apply(lambda x : (x - np.mean(x)) / np.std(x))

	def sigmoid(self, index):
		house_weights = np.array(self.weights_df.iloc[:, index]).reshape(-1, 1)
		return 1 / (1 + np.exp(-np.array(self.sample_df) @ house_weights))
