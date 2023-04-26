import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Plot histogram of each feature to see the distribution of each feature
def plot(df):
	df = pd.read_csv('../../assets/dataset_train.csv', index_col=0)
	features = df.columns[5:].to_list()
	df = df.drop(df.iloc[:, 1:5], axis=1)

	fig = plt.figure(figsize=(12, 9))
	axs = fig.subplots(nrows=3, ncols=5)
	for i, feature in enumerate(features):
		axs[i // 5][i % 5].hist(df[feature].dropna(), bins=100, alpha=0.5)
		axs[i // 5][i % 5].set_title(feature)
		axs[i // 5][i % 5].set_xlabel("Value")
		axs[i // 5][i % 5].set_ylabel("Frequency")

	# Remove empty subplots
	axs[2, 3].axis('off')
	axs[2, 4].axis('off')

	# Put margins between subplots
	plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	plt.show()

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]):
		sys.exit("Error: dataset doesn't exist.")
	try:
		df = pd.read_csv(sys.argv[1], index_col = 0)
	except Exception as e:
		sys.exit(f"Error: {e}")
	plot(df)