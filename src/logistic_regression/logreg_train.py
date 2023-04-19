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

	# Get probability
	def softmax(self):
		exp = np.exp(self.logits)
		self.softmax = exp / np.sum(exp, axis=1, keepdims=True)
		print(self.softmax)
	
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
