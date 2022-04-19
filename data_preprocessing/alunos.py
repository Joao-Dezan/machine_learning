# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:07:58 2022

@author: joaov
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Importing the dataset
dataset = pd.DataFrame([
    ['Student A', 17, 'F', 'Yes'],
    ['Student B', 45, 'M', 'Yes'],
    ['Student C', 13, 'M', 'No'],
    ['Studant D', np.nan, 'F', 'No']],
    columns='nome idade sexo media'.split())

print(dataset)

# Matrix of features (Independent Variables)
X = dataset.iloc[:, 1:-1].values

# Dependent variables' vector
y = dataset.iloc[:, -1].values

#print(X)

# Taking care of missing data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer.fit(X[:, [1,3]])
X[:, [0]] = imputer.fit_transform(X[:, [0]])

#print(X)

# Encoding categorical data
# Using the method: One Hot Encoder


le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])
y = le.fit_transform(y)

#print(X)
#print(y)

# Spliting into two
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print('X_train\n', X_train)
print('X_test\n', X_test)
print('y_train\n', y_train)
print('y_test\n', y_test)

# Feature Scaling
sc = StandardScaler()
X_train[:, 0:1] = sc.fit_transform(X_train[:, 0:1])
X_test[:, 0:1] = sc.transform(X_test[:, 0:1])

print('X_train\n', X_train)
print('X_test\n', X_test)