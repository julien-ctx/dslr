import sys, os
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime

# https://fr.wikipedia.org/wiki/Encodage_one-hot
# https://en.wikipedia.org/wiki/Softmax_function
# https://en.wikipedia.org/wiki/Sigmoid_function

class LogisticRegression:
	def __init__(self):
		self.houses = ['Hufflepuff', 'Gryffindor', 'Ravenclaw', 'Slytherin']
		
	def prepare_training(self, df):
		# Clean dataframe to only keep input features.
		self.sample = df
		self.sample = self.sample.drop('Index', axis=1).drop(self.sample.columns[1:4], axis=1)
		self.convert()
		self.interpolate()
		self.standardize()
		# Bias is added to take into account every value independently from their value.
		self.sample['Bias'] = np.ones(self.sample.shape[0])
		# Weights allows us, for each house to determine the sweep of every feature.
		# It will be used later in our prediction and is set to random non null values for the moment.
		self.weights = np.random.randn(self.sample.shape[1], 1) * 0.01

	def prepare_prediction(self, sample_path, weights_path):
		# Get dataframe of samples (which are the people we want to put in one of the 4 houses).
		self.sample = pd.read_csv(sample_path)
		self.sample = self.sample.drop('Index', axis=1).drop(self.sample.columns[1:4], axis=1)
		self.convert()
		self.interpolate()
		self.standardize()
  
		self.sample['Bias'] = np.ones(self.sample.shape[0])

		# Get weights which will be used to compute the probability with the value of the different features.
		self.weights = pd.read_csv(weights_path)

	# Convert birthday and best hand to integers to take them into account
	def convert(self):
		self.sample['Best Hand'] = self.sample['Best Hand'].replace('Left', 0).replace('Right', 1)
		self.sample['Birthday'] = self.sample['Birthday'].apply(self.birthday_to_age)

	def birthday_to_age(self, birthday):
		birth = datetime.strptime(birthday, '%Y-%m-%d')
		age = relativedelta(datetime.now(), birth).years
		return age

	# We fill NaN and null values with the mean of the column.
	# It is necessary to avoid decreasing data accuracy.
	def interpolate(self):
		for x in range(self.sample.shape[1]):
			for y in range(self.sample.shape[0]):
				if pd.isna(self.sample.iloc[y, x]):
					self.sample.iloc[y, x] = self.sample.iloc[:, x].mean()

	# Standardize data to improve performance and big number issues.
	def standardize(self):
		self.sample = self.sample.apply(lambda x : (x - np.mean(x)) / np.std(x))

	def fit(self, df):
		self.prepare_training(df)
	
		if os.path.exists("weights.csv"):
			os.remove("weights.csv")
		weights_df = pd.DataFrame()

		for house in self.houses:
			y_binary = np.array((df['Hogwarts House'] == house).astype(float))
			y_binary = np.reshape(y_binary, (self.sample.shape[0], 1))
			self.sigmoid(self.weights)
			weights_df = self.gradient_descent(y_binary, house, weights_df)

		weights_df.to_csv('../../assets/weights.csv', index=False)
		print('Weights have been successfully computed and stored in weights.csv in assets folder')

	def gradient(self, y_binary):
		return (self.sample.to_numpy().T @ (self.sigmoid(self.weights) - y_binary)) / self.sample.shape[0]

	def gradient_descent(self, y_binary, house, weights):
		base_filename = 'eval_'
		n_files = len([f for f in os.listdir('../../assets') if os.path.isfile(f) and f.startswith(base_filename)])
		eval_file = open(f'../../assets/{base_filename}{n_files}.csv', 'w+')

		alpha = 0.001
		for _ in range(10000):
			self.weights = self.weights - alpha * self.gradient(y_binary)
			if _ % 100 == 0:
				probabilities = self.sigmoid(self.weights)
				loss = -np.mean(y_binary * np.log(probabilities) + (1 - y_binary) * np.log(1 - probabilities))
				# Print or store the loss, for example, appending it to a list
				eval_file.write(f'{loss}\n')

		return pd.concat([weights, pd.DataFrame(self.weights, columns=[house])], axis=1)

	# Get probability with sigmoid function
	def sigmoid(self, weights):
		return 1 / (1 + np.exp(-np.array(self.sample) @ weights))	

	# Find house with the maximum probability
	def find_house(self, i, probs):
		sample_probs = [probs[j][i][0] for j in range(len(probs))]
		max_prob = max(sample_probs)
		house = sample_probs.index(max_prob)
		return self.houses[house]
