import pandas as pd

df = pd.read_csv("./Thyroid_Diff.csv")
print(df.head())

print(df.isnull().sum().sum())