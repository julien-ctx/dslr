import sys, os
import pandas as pd
import numpy as np

class LogisticRegression:

	def __init__(self, df):
		self.df = df
		self.clean_data()
		self.standardize()
		print(self.df)
	
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
