#!/usr/bin/env python3

# import necessary libraries
import pandas as pd
import numpy as np

# load dataset
titanic = pd.read_csv('./datasets/titanic.csv')

# we'll be working with 5 features and one target variable
get_vars = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# subset the dataset
titanic = titanic[get_vars]

# data preprocessing: 
# fill the null values in 'Age' column with the average age
titanic['Age'] = round(titanic['Age'].fillna(titanic['Age'].mean()))

# fill the embarked column with the most frequent value
titanic['Embarked'] = titanic['Embarked'].fillna('S')

# We've finished dealing with missing values. Next, we'll use one-hot-encoding to convert the categorical columns to numeric ones.
titanic = pd.get_dummies(titanic, columns = ['Sex', 'Embarked'], drop_first = True)

# get the features
X = titanic.drop(['Survived'], axis = 1)

# get the target
Y = titanic['Survived']

# fit a logistics regression
# create a linear model
from sklearn.linear_model import LogisticRegression

logreg_model = LogisticRegression(max_iter = 10000)

logreg_model.fit(X, Y)

print("----Training Model-------")

# saving the model
from joblib import dump, load
dump(logreg_model, 'model.pkl')

print("Model dumped!")

# Load the model that you just saved
loaded_model = load('model.pkl')

# Save the features columns 
model_columns = list(X.columns)

# save model columns
dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")