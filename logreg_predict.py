import numpy as np
import sys
import pandas as pd

if __name__ == '__main__':
    try:
        "Read file to predict"
        df = pd.read_csv(sys.argv[1])
        "Load weights"
        weights = np.load('Weights.npy',allow_pickle=True)
    except:
        print("Usage: python3 logreg_predict.py filename.csv weights_file.npy")
        exit (-1)