import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

houses = ["Gryffindor", "Slytherin", "Ravenclaw", "Hufflepuff"]
color = {
    "Gryffindor": "red",
    "Slytherin": "green",
    "Ravenclaw": "blue",
    "Hufflepuff": "yellow"
}

df = pd.read_csv('../../assets/dataset_train.csv')
df.dropna(axis=0, inplace=True)
df_houses = {
    "Gryffindor": df[df["Hogwarts House"] == "Gryffindor"],
    "Slytherin": df[df["Hogwarts House"] == "Slytherin"],
    "Ravenclaw": df[df["Hogwarts House"] == "Ravenclaw"],
    "Hufflepuff": df[df["Hogwarts House"] == "Hufflepuff"]
}

for h in houses:
    plt.scatter(df_houses[h]["Defense Against the Dark Arts"], df_houses[h]["Astronomy"], color=color[h], label=h, alpha=0.5)
plt.title("Defense Against the Dark Arts vs Astronomy")
plt.xlabel("Defense Against the Dark Arts")
plt.ylabel("Astronomy")

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in color.values()]
plt.legend(handles, houses)
plt.show()


