import pandas as pd
import numpy as np
import sys, os

from describe import mean, std

# Skewness: measure of the asymmetry of the probability distribution of a real-valued random variable
# Skewness = 0: symmetric distribution.
# Skewness > 0: more weight in the left tail of the distribution
# Skewness < 0: more weight in the right tail of the distribution
def skewness(x):
    return np.nansum((x - mean(x)) ** 3) / (x.shape[0] * std(x) ** 3)

# Kurtosis: measure of whether the data are heavy-tailed or light-tailed relative to a normal distribution
# Kurtosis = 3: mesokurtic. Normal distribution.
# Kurtosis > 3: leptokurtic. Heavier tails than a normal distribution.
# Kurtosis < 3: platykurtic. Lighter tails than a normal distribution.
def kurtosis(x):
	return np.nansum((x - mean(x)) ** 4) / (x.shape[0] * std(x) ** 4)

# Mean absolute deviation: average of the absolute deviations of the values from their mean
def mad(x):
	return np.nansum(np.abs(x - mean(x))) / x.shape[0]

# Coefficient of variation: ratio of the standard deviation to the mean
def cv(x):
	return std(x) / mean(x)

def get_description(col):
	col = col[~pd.isna(col)]
	return skewness(col), kurtosis(col), mad(col), cv(col)

def describe(dataset):
	return pd.DataFrame(data = np.apply_along_axis(get_description, 0, np.array(dataset)[:,6:]), 
		index = ["Skewness", "Kurtosis", "MAD", "CV"],
		columns = dataset.columns.values[6:])

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")
	print(describe(df))
