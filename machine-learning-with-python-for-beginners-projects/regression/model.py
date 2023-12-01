import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def transform(data):
    # Data preprocessing
    num_col = data.columns[data.columns != "Category"]
    cat_col = ["Category"]

    num_preprocessing = Pipeline([
        ('imp', SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    full_preprocessing = ColumnTransformer([
        ('num', num_preprocessing, num_col),
        ('cat', OneHotEncoder(
            dtype=int, sparse_output=False, drop='first'), cat_col)
    ])

    return full_preprocessing.fit_transform(data)

def predict_and_print_metrics(lr, X, y):
    y_predicted = lr.predict(X)

    r2 = r2_score(y, y_predicted)
    RMSE = mean_squared_error(y, y_predicted, squared=False)
    print(f"RMSE={RMSE} r2={r2}")
    return y_predicted

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

# The description and histogram say that "price" and "DSTLU" 
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

# Training the model
X_train_transformed = transform(X_train) 
model =  LinearRegression()
model.fit(X_train_transformed, y_train)

print("Coefficients of the model (a0, [a1, ..., an]):")
print(model.intercept_, model.coef_)

# Evaluating on the training set
print("Metrics on the training set")
y_train_pred = predict_and_print_metrics(model, X_train_transformed, y_train)

# Evaluating on the test set
print("Metrics on the test set")
X_test_transformed = transform(X_test)
predict_and_print_metrics(model, X_test_transformed, y_test)
