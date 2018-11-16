import sys
sys.path.append('../')
import numpy as np
import pandas as pd
from math import sqrt
from utils.utils import *
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.linear_model import Lasso, LassoCV, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from scipy.stats import skew
from scipy.special import boxcox1p

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
from utils.transform import *
pd.options.mode.chained_assignment = None

from sklearn.svm import SVC
from sklearn.svm import SVR


"""
load data
"""
train_set = pd.read_csv('../data/train.csv')
test_set = pd.read_csv('../data/test.csv')

"""
Remove Outliers
"""
outliers = train_set[ train_set['GrLivArea'] > 4500 ].index
#print(outliers)

outliers = [197, 523, 691, 854, 1182, 1298]


train_set.drop(outliers, inplace=True)

"""
fix salePrice skewness
"""
train_set["SalePrice"] = np.log1p(train_set["SalePrice"])
y_train_values = train_set["SalePrice"].values

"""
prepare combined data.
"""
train_set_id = train_set['Id']
test_set_id = test_set['Id']

train_set_rows = train_set.shape[0]
test_set_rows = test_set.shape[0]

train_set.drop('Id', axis=1, inplace=True)
test_set.drop('Id', axis=1, inplace=True)
train_set.drop('SalePrice', axis=1, inplace=True)

combined_data = pd.concat((train_set, test_set))

nansum = combined_data.isnull().sum()
nansum = pd.DataFrame({'col': nansum.index, 'sum': nansum.values}).sort_values(by='sum', ascending=False)
nansum = nansum[nansum['sum'] > 0]


X = combined_data


X.loc[X[(X['PoolArea'] > 0) & (X['PoolQC'].isnull())].index,['PoolQC']] = 'TA'


#print(X['PoolQC'].isnull().index)

X.loc[X['PoolQC'].isnull().index, 'PoolQC'] = 'NO'

tmp = X[X['PoolQC'].isnull()]
#pd.set_option('max_columns', None)
#print(X)


#X.loc[X['MiscFeature'].isnull().index, 'MiscFeature'] = 'NO'
#print("@@@")
#print(X.loc[X[X['FireplaceQu'].isnull()].index, 'FireplaceQu'])

#print(X[X['FireplaceQu'].isnull()])
#1420

#X.loc[X['FireplaceQu'].isnull(), 'FireplaceQu'] = 'NA'
#print(X[X['FireplaceQu'].isnull()])
#print(X[X['FireplaceQu'].isnull()]['FireplaceQu'])



X.loc[X['GarageType'].isnull(), ['GarageFinish','GarageQual','GarageCond','GarageYrBlt','GarageType','GarageCars','GarageArea']] \
    = ['NA','NA','NA',0,'NA',0,0]


# GarageFinish  NA
#GarageQual   NA
#GarageCond   NA
#GarageYrBlt   0
#GarageType   NA
# GarageCars     0
#GarageArea     0

t=X.loc[(X['GarageCond'].isnull()) & (X['OverallCond'] == 8),['GarageFinish','GarageQual','GarageCond','GarageYrBlt']]

X.loc[(X['GarageCond'].isnull()) & (X['OverallCond'] == 8),
      ['GarageFinish','GarageQual','GarageCond','GarageYrBlt']]\
   = [
       X[X['OverallCond']==8]['GarageFinish'].mode()[0],
       X[X['OverallCond']==8]['GarageQual'].mode()[0],
       X[X['OverallCond']==8]['GarageCond'].mode()[0],
       1910,
   ]

X.loc[(X['GarageCond'].isnull()) & (X['OverallCond'] == 6),
      ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageCars', 'GarageArea']] \
    = [
    X[X['OverallCond'] == 6]['GarageFinish'].mode()[0],
    X[X['OverallCond'] == 6]['GarageQual'].mode()[0],
    X[X['OverallCond'] == 6]['GarageCond'].mode()[0],
    1923,
    X[X['OverallCond'] == 6]['GarageCars'].median(),
    X[X['OverallCond'] == 8]['GarageArea'].median()
]