import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

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