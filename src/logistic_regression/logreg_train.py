import sys, os
import pandas as pd

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists('../../assets/dataset_train.csv'):
		sys.exit("Error: dataset doesn't exist.")
	df = pd.read_csv(sys.argv[1])
