import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

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

# Splitting
y = df["Rating"]
X = df.loc[:, df.columns != "Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

print("Correlation table: ")
print(X_train.loc[:, X_train.columns != "Category"].corr())

for col in X_train:
    x = df[col]
    plt.scatter(x, y)
    plt.savefig(f"scatter_plots/{col}-plot.png")
    plt.clf()

# Data preprocessing
num_col = X.columns[X.columns != "Category"]
cat_col = ["Category"]

imp = SimpleImputer(strategy="mean")
tf_num = imp.fit_transform(X_train[num_col])

scaler = StandardScaler()
tf_num = scaler.fit_transform(tf_num)

ohe = OneHotEncoder(dtype=int, sparse_output=False, drop='first')
tf_cat = ohe.fit_transform(X_train[cat_col])

X_train_transformed = np.concatenate((tf_num, tf_cat), axis=1)

# Training the model
model =  LinearRegression()
model.fit(X_train_transformed, y_train)

print("Coefficients of the model (a0, [a1, ..., an]):")
print(model.intercept_, model.coef_)