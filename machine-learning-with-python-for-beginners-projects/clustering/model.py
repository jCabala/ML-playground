import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.data
y = digits.target

# Splitting the set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=0)

# Elbow point method (finding the optimal number of clusters)
MIN_CLUSTERS, MAX_ClUSTERS = 10, 50
cluster_numbers = range(MIN_CLUSTERS, MAX_ClUSTERS, 5)
inertias = []

for i in cluster_numbers:
    kmeans = KMeans(n_init=10, n_clusters=i)
    kmeans.fit(X_train)
    inertias.append(kmeans.inertia_)

# Plotting the inertias
plt.figure(figsize=(12, 4))
plt.grid()
plt.plot(cluster_numbers, inertias)
plt.xticks(cluster_numbers)
plt.xlabel("Num of clusters")
plt.ylabel('Inertia')
plt.savefig("./elbow_point.png")

num_of_clusters = MIN_CLUSTERS
best_inertia_change = 0 

for i in range(1, len(cluster_numbers)):
    intertia_change = inertias[i - 1] - inertias[i]
    if intertia_change > best_inertia_change:
        num_of_clusters = cluster_numbers[i]
        best_inertia_change = intertia_change

pipeline = Pipeline([
    ('cluster', KMeans(n_init=10, random_state=0, n_clusters=num_of_clusters)),
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=0))])

# Searching for the best params
params = {'svm__C': [1, 5, 8, 10], 
          'svm__kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
grid_search = GridSearchCV(pipeline, params)
grid_search.fit(X_train, y_train)
 
print(f"""Grid search completed.
Best score was: {grid_search.best_score_},
generated by params: {grid_search.best_params_}""") 

# Evaluating the model
model = grid_search.best_estimator_
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy of the best estimator was {accuracy}")