from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
X = iris.data.features 
y = iris.data.targets 

# combine features and targets into one dataframe to check duplicates
df = pd.concat([X, y], axis=1)

duplicates = df.duplicated()
print(duplicates)
print(f"Number of duplicate rows: {duplicates.sum()}")
print(df[duplicates])

# metadata
print(iris.metadata)

# variable information
print(iris.variables)

# number of features
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")

# number of classes
print(f"Number of classes: {y.nunique().iloc[0]}")
print(f"Class names: {y.iloc[:, 0].unique()}")

# records per class
print(y.iloc[:, 0].value_counts())

