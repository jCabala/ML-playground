import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

# --- Reading data ---
df = pd.read_csv("./Thyroid_Diff.csv")

# Splitting the dataset
X = df.loc[:, df.columns != 'Recurred']
y = df['Recurred']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=36)

# --- Data preprocessing ---

# Sorting the columns into categorical and numerical
numericalColumns, categoricalColumns = [], []

for col in X_train:
    entry = df[col][0]
    if type(entry) == str:
        categoricalColumns.append(col)
    else:
        numericalColumns.append(col)


# Transforming columns 
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(dtype=int, sparse_output=False, drop='first'), 
    categoricalColumns),
    ('scalar', StandardScaler(), numericalColumns)
    ])


# Creating the pipeline
pipeline = Pipeline([
    ('ct', ct)])

# Printing the results
np.set_printoptions(threshold=np.inf)
print(pipeline.fit_transform(X_train))