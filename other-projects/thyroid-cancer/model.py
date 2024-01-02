import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

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

for col in X:
    entry = df[col][0]
    if type(entry) == str:
        categoricalColumns.append(col)
    else:
        numericalColumns.append(col)


# Transforming columns 
ct = ColumnTransformer([
    ('ohe', OneHotEncoder(dtype=int, handle_unknown="ignore", 
                          sparse_output=False, drop='first'), 
    categoricalColumns),
    ('scalar', StandardScaler(), numericalColumns)
    ])


# --- Dimensional Reduction ---
pca = PCA(n_components=0.9)

# --- Model 1: SVM ---
svm = SVC(random_state=0)

# Creating the pipeline
svm_pipeline = Pipeline([
    ('ct', ct),
    ('pca', pca),
    ('svm', svm)
    ])

# Grid search
params = {'svm__C': [1, 5, 8, 10],
          'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid_search = GridSearchCV(svm_pipeline, params)
grid_search.fit(X_train, y_train)

svm_estimator = grid_search.best_estimator_

# Evaluating accuracy on the test set
y_pred_train = svm_estimator.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train) 

y_pred_test = svm_estimator.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"Accuracy of the SVM model on the train set is: {accuracy_train}" 
      + f"and on a test set is: {accuracy_test}")