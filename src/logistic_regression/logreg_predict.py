import sys, os
import pandas as pd
import numpy as np
from logreg import LogisticRegressionPredict

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
		sys.exit("Error: dataset or weights file doesn't exist.")

	model = LogisticRegressionPredict(sys.argv[1], sys.argv[2])

	houses = ['Hufflepuff', 'Gryffindor', 'Ravenclaw', 'Slytherin']
	for index, house in enumerate(model.weights_df.columns):
		print(model.sigmoid(index))
