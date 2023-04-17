import sys, os
import pandas as pd

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists('../../assets/dataset_train.csv') or not os.path.exists('weights.csv'):
		sys.exit("Error: dataset or weights file doesn't exist.")
