import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data
y = digits.target

# Displaying the first instance as an image
some_digit = X[8].reshape(8,8)
plt.imshow(some_digit, cmap="gray")
plt.savefig("./some_digit.png")
plt.clf()