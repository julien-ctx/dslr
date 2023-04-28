import numpy as np
import sys, os
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
sys.path.append('../logistic_regression')
from logreg import LogisticRegression 

if __name__ == "__main__":
	if not os.path.exists('../../assets/dataset_train.csv'):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv('../../assets/dataset_train.csv')
	except Exception as e:
		sys.exit(f"Error: {e}")

kf = KFold(n_splits=5)

total_accuracy = []
for train_i, val_i in kf.split(df):
	model = LogisticRegression()
	model.fit(df.iloc[min(train_i):max(train_i), :], 'Default')
	
	tmp_predict = df.iloc[min(val_i):max(val_i) + 1, :]
	tmp_predict = tmp_predict.drop('Hogwarts House', axis=1)
	tmp_predict.insert(loc=1, column='Hogwarts House', value='')
	tmp_predict.to_csv('../../assets/tmp_predict.csv', index=False)
	os.system('python3 ../logistic_regression/logreg_predict.py ../../assets/tmp_predict.csv ../../assets/weights.csv')

	prediction_df = pd.read_csv('../../assets/houses.csv')
	accuracy = round(100 * accuracy_score(df.loc[min(val_i):max(val_i), 'Hogwarts House'], prediction_df['Hogwarts House']), 4)
	total_accuracy.append(accuracy)
	print(f"Accuracy on validation dataset: {accuracy}%\n")

print(f"Average accuracy: {round(np.mean(total_accuracy), 4)}%")
