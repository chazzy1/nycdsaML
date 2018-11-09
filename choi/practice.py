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



test = [1,2,3,4]

print(list(map(lambda x: x + 1 , test )))