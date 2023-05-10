import sys, os
import pandas as pd
import numpy as np
from logreg import LogisticRegression

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Error: wrong parameter number.\nUsage: python3 logreg_predict.py [dataset_test.csv] [weights.csv]")
	if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
		sys.exit("Error: dataset or weights file doesn't exist.")

	model = LogisticRegression()
	model.prepare_prediction(sys.argv[1], sys.argv[2])

	probs = []
	house_results = []
	# Get probabilities of samples for each house, thanks to sigmoid function and weights.
	for index, house in enumerate(model.weights.columns):
		probs.append(model.sigmoid(model.sample.to_numpy(), np.array(model.weights.iloc[:, index]).reshape(-1, 1)).tolist())
	# Find the house with the maximum probability and store it to create houses.csv later.
	for i in range(model.sample.shape[0]):
		house_results.append(model.find_house(i, probs))
	
	# Create result output.
	results_df = pd.DataFrame(house_results, columns=['Hogwarts House'])
	results_df.insert(0, 'Index', results_df.index)
	results_df.to_csv('../../assets/houses.csv', index=False)
	print('Results have been successfully stored in houses.csv in assets folder.')
