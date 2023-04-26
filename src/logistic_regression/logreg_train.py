import sys, os
import pandas as pd
import numpy as np
from logreg import LogisticRegressionTrain

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1])
	except Exception as e:
		sys.exit(f"Error: {e}")

	model = LogisticRegressionTrain(df)
	model.preprocess_data()
	
	if os.path.exists("weights.csv"):
		os.remove("weights.csv")
	weights_df = pd.DataFrame()

	houses = ['Hufflepuff', 'Gryffindor', 'Ravenclaw', 'Slytherin']
	for house in houses:
		y_binary = np.array((df['Hogwarts House'] == house).astype(float))
		y_binary = np.reshape(y_binary, (1600, 1))
		model.sigmoid()
		weights_df = model.gradient_descent(y_binary, house, weights_df)

	weights_df.to_csv('../../assets/weights.csv', index=False)
