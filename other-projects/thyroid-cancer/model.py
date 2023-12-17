import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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

for col in df:
    entry = df[col][0]
    print(col, type(entry))
    if type(entry) == str:
        categoricalColumns.append(col)
    else:
        numericalColumns.append(col)

# Categorical data encoding 
cat_ct = ColumnTransformer([(
    'ohe', 
    OneHotEncoder(dtype=int, sparse=False, drop='first'), 
    categoricalColumns)])


# Creating the pipeline
pipeline = Pipeline(('cat_encoding', cat_ct))
