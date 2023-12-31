# -*- coding: utf-8 -*-
"""K-Nearest-Neighbors-From-Scratch.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IIcVZQIlQ-FaQTgrsXjP8NS185ETFaPh

# KNN Estimator
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Asm dim(p) == dim(q)
def euc_dist(_p, _q):
  p,q = np.array(_p), np.array(_q)
  return np.sqrt(np.sum((p - q)**2))

class KNN:
  def __init__(self, k=3):
    self.k = k
    self.points = None

  def fit(self, points):
    self.points = points

  def predict(self, new_points):
    return list(map(self.__predict_one, new_points))

  def __predict_one(self, new_point):
    distances = []

    for category in self.points:
      for point in self.points[category]:
        distance = euc_dist(point, new_point)
        distances.append([distance, category])

    categories = [category[1]
                  for category in sorted(distances)[:self.k]]

    return Counter(categories).most_common(1)[0][0]

"""# 2D KNN Example

"""

bluePoints = np.random.rand(20, 2) * 50
redPoints = np.random.rand(20, 2) * 50

testPoints = np.random.rand(20, 2) * 50
trainPoints = {"blue": bluePoints, "red": redPoints}

def scatter_train():
  plt.clf()
  plt.scatter(bluePoints[:, 0], bluePoints[:, 1], color="blue")
  plt.scatter(redPoints[:, 0], redPoints[:, 1], color="red")

scatter_train()
plt.scatter(testPoints[:, 0], testPoints[:, 1], color="green")
plt.show()

knn = KNN(k=4)
knn.fit(trainPoints)
pred_labels = knn.predict(testPoints)

zipPoints = list(zip(testPoints, pred_labels))

def get_points_by_cat(cat):
  return list(map(lambda pt : pt[0],
              filter(lambda a : a[1] == cat, zipPoints)))

pred_red_points = np.array(get_points_by_cat('red'))
pred_blue_points = np.array(get_points_by_cat('blue'))

scatter_train()
plt.scatter(pred_red_points[:, 0], pred_red_points[:, 1],
            color="red", marker="x")
plt.scatter(pred_blue_points[:, 0], pred_blue_points[:, 1],
            color="blue", marker="x")
plt.show()