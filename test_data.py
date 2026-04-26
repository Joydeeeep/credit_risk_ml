import pandas as pd

df = pd.read_csv("data/credit_risk_dataset.csv")

print(df.shape)
print(df.columns.tolist())
print(df.head())