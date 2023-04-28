import sys, os
import pandas as pd
import numpy as np
from logreg import LogisticRegression
from utils import Timer

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.\nUsage: python3 logreg_train.py [dataset_train.csv]")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	mode = input("Choose a logistic regression mode to train the model:\n1 - Default\n2 - Stochastic\n3 - Mini-batch\n")
	if mode not in ['1', '2', '3']:
		sys.exit("Error: wrong mode.")
	
	timer = Timer()
	timer.tic('Training model...\r')

	model = LogisticRegression(int(mode))
	model.fit(df)

	timer.toc('Training completed.')
