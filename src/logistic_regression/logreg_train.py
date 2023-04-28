import sys, os
import pandas as pd
import numpy as np
from logreg import LogisticRegression
from utils import Timer

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Error: wrong parameter number.\nUsage: python3 logreg_train.py [dataset_train.csv] [Default/Stochastic]")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	timer = Timer()
	timer.tic('Training model...\r')

	model = LogisticRegression()
	model.fit(df, sys.argv[2])

	timer.toc('Training completed.')
