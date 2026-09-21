import pandas as pd
pd.set_option("display.max_columns", None)
df = pd.read_csv("data/loans.csv")

print(df.shape)
print(df.head())
print(df["LoanStatus"].value_counts())
print(df["FirstDisbursementDate"].head())
print(df["ChargeOffDate"].head())
