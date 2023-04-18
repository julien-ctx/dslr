import pandas as pd
import numpy as np
import math, sys, os

def mean(x):
	return np.nansum(x) / get_valid_count(x)

def median(x):
	valid_count = get_valid_count(x)
	return x[valid_count // 2] if valid_count % 2 == 1 else (x[valid_count // 2 - 1] + x[valid_count // 2]) / 2

def quartiles(x):
	valid_count = get_valid_count(x)
	return median(x[:valid_count // 2 + (1 if valid_count % 2 == 1 else 0)]), median(x), median(x[valid_count // 2:])

def std(x):
	valid_count = get_valid_count(x)
	x = (x - mean(x)) ** 2
	return math.sqrt(np.nansum(x) / valid_count)

def get_valid_count(x):
	return x.shape[0] - np.sum(pd.isnull(x))

def get_description(a):
	x = np.sort(a)
	tmp = quartiles(x)
	return get_valid_count(x), mean(x), std(x), tmp[0], tmp[1], tmp[2], x[0], x[-1]

def describe(dataset):
	return pd.DataFrame(data = np.apply_along_axis(get_description, 0, np.array(dataset)[:,6:]), 
		index = ["Count", "Mean", "Std", "25%", "50%", "75%", "Min", "Max"],
		columns = dataset.columns.values[6:])

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1]).dropna()
	except Exception as e:
		sys.exit(f"Error: {e}")
	print(describe(df))
	
	# Compare with the one of Pandas lib
	# print(df.iloc[:, 5:].describe())
