import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

digits = load_digits()
print(type(digits))
print(type(digits.data))
print(digits.data.shape)