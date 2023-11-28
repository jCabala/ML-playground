import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# Plotting histograms for each column
df = pd.read_csv('reviews.csv')
for col in df:
    if col == "Category":
        continue
    
    s = df[col]
    ax = s.hist()
    fig = ax.get_figure()
    fig.savefig(f"histograms/{col}-hist.png")
    plt.clf()

# Saving the data description
print(df.describe(), file=open("data_description.txt", "w"))

# The description and histogram say that "prica" and "DSTLU" 
# columns have negarive values. That is likely an error.
numInvalid = (len(df[df["Days since Last Update"] < 0]) + 
              len(df[df["Price"] < 0]))
print(f"Num of invalid entries is {numInvalid}")

# Since there are not many invalid rows, let's drop them
df = df.drop(df[(df["Price"] < 0) | (df["Days since Last Update"] < 0)].index)
print(f"After dropping invalid entries we get a shape of {df.shape}")