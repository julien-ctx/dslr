import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
color = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Ravenclaw": "blue",
    "Hufflepuff": "yellow"
}

df = pd.read_csv('../../assets/dataset_train.csv')
df_houses = {
    "Gryffindor": df[df["Hogwarts House"] == "Gryffindor"].drop(df.iloc[:, :6], axis = 1),
    "Slytherin": df[df["Hogwarts House"] == "Slytherin"].drop(df.iloc[:, :6], axis = 1),
    "Ravenclaw": df[df["Hogwarts House"] == "Ravenclaw"].drop(df.iloc[:, :6], axis = 1),
    "Hufflepuff": df[df["Hogwarts House"] == "Hufflepuff"].drop(df.iloc[:, :6], axis = 1)
}
df = df.drop(df.iloc[:, :6], axis = 1)

fig, axs = plt.subplots(3, 5)
fig.tight_layout()
for i, col in enumerate(df.columns):
    for h in houses:
        axs[i // 5][i % 5].hist(df_houses[h][col].dropna(), bins=20, color=color[h], label=h, alpha=0.5)
    axs[i // 5][i % 5].set_title(col)
plt.margins(3)
handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in color.values()]
plt.legend(handles, houses)
plt.show()

