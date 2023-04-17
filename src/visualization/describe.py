import pandas as pd
import numpy as np
import math

def mean(x):
	return np.nansum(x) / x.shape[0]

def median(x):
	return x[x.shape[0] // 2] if x.shape[0] % 2 == 1 else (x[x.shape[0] // 2 - 1] + x[x.shape[0] // 2]) / 2

def quartiles(x):
	return median(x[:x.shape[0] // 2 + (1 if x.shape[0] % 2 == 1 else 0)]), median(x), median(x[x.shape[0] // 2:])

def std(x):
	x = (x - mean(x)) ** 2
	return math.sqrt(np.nansum(x) / x.shape[0])

def get_description(a):
	x = np.sort(a)
	tmp = quartiles(x)
	return x.shape[0], mean(x), std(x), tmp[0], tmp[1], tmp[2], x[0], x[x.shape[0] - 1]

def describe(dataset):
	return pd.DataFrame(data = np.apply_along_axis(get_description, 0, np.array(dataset)[:,6:]), 
		index = ["Count", "Mean", "Std", "25%", "50%", "75%", "Min", "Max"],
		columns = dataset.columns.values[6:])

df = pd.read_csv('../../assets/dataset_train.csv')
print(describe(df))
