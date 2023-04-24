import sys, os
import pandas as pd

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Error: wrong parameter number.")
	if not os.path.exists(sys.argv[1]) or not os.path.exists(sys.argv[2]):
		sys.exit("Error: dataset or weights file doesn't exist.")
	