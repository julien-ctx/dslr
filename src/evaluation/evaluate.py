import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit("Error: wrong parameter number.")
    if not os.path.exists(sys.argv[1]):
        sys.exit("Error: dataset doesn't exist.")
    df = pd.read_csv(sys.argv[1])
    arr = np.array(df)

    plt.plot(arr)

    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss Progress')

    plt.show()