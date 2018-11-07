import numpy as np
import pandas as pd

from scipy.special import boxcox1p, boxcox

"""
load data
"""
train = pd.read_csv('./data/train.csv')


"""
fix salePrice skewness
"""
train["SalePrice"] = np.log1p(train["SalePrice"])



y_train_values = train.SalePrice.values

all_features_data = train

all_features_data.drop(['SalePrice'], axis=1, inplace=True)

"""
fix NaN
"""
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType',
            'MSZoning', 'Functional', 'Electrical','KitchenQual', 'Exterior1st',
            'Exterior2nd', 'SaleType', 'MSSubClass'
            ]:
    all_features_data[col] = all_features_data[col].fillna(all_features_data[col].mode()[0])


for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
            'Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType',
            'LotFrontage'
            ]:
    all_features_data[col] = all_features_data[col].fillna(0)

"""
encode categorical features
"""
from sklearn.preprocessing import LabelEncoder
cols_encoding_needed = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')


for col in cols_encoding_needed:
    lbl = LabelEncoder()
    lbl.fit(list(all_features_data[col].values))
    all_features_data[col] = lbl.transform(list(all_features_data[col].values))


"""
fix numeric features skewness by applying boxcox
"""
numeric_columns = all_features_data.dtypes[all_features_data.dtypes != "object"].index

from scipy.stats import skew
skewed_columns = all_features_data[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewed_columns.index

for feat in skewed_features:
    all_features_data[feat] = boxcox1p(all_features_data[feat], 0.15)


all_features_data = pd.get_dummies(all_features_data)



"""
lasso fit
"""
from sklearn.linear_model import Lasso
lasso = Lasso()

lasso.set_params(alpha=0.0005, normalize=True)

model = lasso.fit(all_features_data.values, y_train_values)

from sklearn.metrics import mean_squared_error
from math import sqrt

y_pred = model.predict(all_features_data)

print("Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, y_pred)))