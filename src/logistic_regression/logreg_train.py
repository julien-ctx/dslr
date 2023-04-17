import sys, os
import pandas as pd
import numpy as np

class LogisticRegression:
	@staticmethod
	def clean_data(df):
		new_df = df.dropna()
		new_df.loc[new_df['Best Hand'] == 'Left', 'Best Hand'] = 0.0
		new_df.loc[new_df['Best Hand'] == 'Right', 'Best Hand'] = 1.0
		return new_df.iloc[:, 5:]

	@staticmethod
	def standardize(x):
		return (x - np.mean(x)) / np.std(x)

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists('../../assets/dataset_train.csv'):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	# Drop NaN values, replace Left and Right by boolean values in order to make them count
	reg_df = LogisticRegression.clean_data(df)
	reg_df = reg_df.apply(LogisticRegression.standardize)
	print(reg_df)
