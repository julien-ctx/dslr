import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
color = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Ravenclaw": "blue",
    "Hufflepuff": "yellow"
}

df = pd.read_csv('../assets/dataset_train.csv')
features_to_drop = ['Index', 'First Name', 'Last Name', 'Birthday', 'Best Hand']
df.drop(features_to_drop, axis=1, inplace=True)
df = pd.DataFrame.dropna(df, axis=0)
sns.pairplot(df, hue='Hogwarts House', palette=color)
plt.show()