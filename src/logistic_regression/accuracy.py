import sys, os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
	real_df = pd.read_csv('../../assets/dataset_train.csv')
	real_df = real_df.drop(real_df.columns[2:], axis=1)

	to_predict_df = pd.read_csv('../../assets/dataset_train.csv')
	to_predict_df = to_predict_df.drop('Hogwarts House', axis=1)
	to_predict_df.insert(loc=1, column='Hogwarts House', value='')

	to_predict_df.to_csv('../../assets/to_predict.csv', index=False)
	
	os.system('python3 logreg_train.py ../../assets/dataset_train.csv')
	os.system('python3 logreg_predict.py ../../assets/to_predict.csv ../../assets/weights.csv')

	prediction_df = pd.read_csv('../../assets/houses.csv')
	print(f"Accuracy on training dataset: {100 * accuracy_score(real_df['Hogwarts House'], prediction_df['Hogwarts House'])}%")
