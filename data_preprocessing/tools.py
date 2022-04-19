# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:47:34 2022

@author: joaov
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Importing the dataset
dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(dataset)
#print(X)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)


# Encoding the Independent Variable
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)

#print(X)

# Encoding the Dependent Variable
le = LabelEncoder()

y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('X_train\n', X_train)
#print('X_test\n', X_test)
#print('y_train\n', y_train)
#print('y_test\n', y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print('\nX_train\n', X_train)
print('\nX_test\n', X_test)