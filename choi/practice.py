import pandas as pd
import numpy as np

"""
load data
"""
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

#print(train.shape)
#print(test.shape)

#print(train.head())
#print(test.head())


all_data = pd.concat((train,test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    print(column, all_data[column].unique())

all_data = pd.concat((train,test))
for column in all_data.select_dtypes(include=[np.object]).columns:
    train[column] = train[column].astype('category', categories = all_data[column].unique())
    test[column] = test[column].astype('category', categories = all_data[column].unique())


X_train = pd.get_dummies(train)
X_test = pd.get_dummies(test)

print(train.shape)
print(test.shape)
print(X_train.shape)
print(X_test.shape)
