import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso
from scipy.stats import skew
import xgboost as xgb

"""
load data
"""
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
"""
fix salePrice skewness
"""
train["SalePrice"] = np.log1p(train["SalePrice"])

y_train_values = train.SalePrice.values

all_features_data = train
all_features_data.drop(['SalePrice'], axis=1, inplace=True)

all_features_test = test

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
    all_features_test[col] = all_features_test[col].fillna(all_features_test[col].mode()[0])

for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
            'Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType',
            'LotFrontage'
            ]:
    all_features_data[col] = all_features_data[col].fillna(0)
    all_features_test[col] = all_features_test[col].fillna(0)

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
    lbl.fit(list(all_features_test[col].values))
    all_features_test[col] = lbl.transform(list(all_features_test[col].values))




"""
fix numeric features skewness by applying boxcox
"""
numeric_columns = all_features_data.dtypes[all_features_data.dtypes != "object"].index
skewed_columns = all_features_data[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]

numeric_columns_test = all_features_test.dtypes[all_features_test.dtypes != "object"].index
skewed_columns_test = all_features_test[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_columns_test = skewed_columns_test[abs(skewed_columns_test) > 0.75]


from scipy.special import boxcox1p
skewed_features = skewed_columns.index
skewed_features_test = skewed_columns_test.index

for feat in skewed_features:
    all_features_data[feat] = boxcox1p(all_features_data[feat], 0.15)

for feat in skewed_features_test:
    all_features_test[feat] = boxcox1p(all_features_test[feat], 0.15)



""""" One-hot encode """

all_data = pd.concat((all_features_data,all_features_test))

for column in all_data.select_dtypes(include=[np.object]).columns:
    all_features_data[column] = all_features_data[column].astype('category', categories = all_data[column].unique())
    all_features_test[column] = all_features_test[column].astype('category', categories = all_data[column].unique())

all_features_data = pd.get_dummies(all_features_data)
all_features_test = pd.get_dummies(all_features_test)


"""
lasso fit
"""
lasso = Lasso()
model = lasso.fit(all_features_data.values, y_train_values)

y_pred = model.predict(all_features_data)

''' find valid lasso columns '''
model_coef = model.coef_.tolist()

model_matrix = np.concatenate((all_features_data.columns.values.reshape(-1,1),
                               model.coef_.reshape(-1,1)), axis=1)
model_matrix = pd.DataFrame.from_records(model_matrix)
model_matrix.columns = ['feature','coef']

model_matrix = model_matrix[model_matrix['coef'] != 0 ]
valid_col = model_matrix['feature'].values

""" xgboost """

# gb = xgb.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
#                            colsample_bytree=1, max_depth=7)
#
# model_gb = gb.fit(valid_col, y_train_values)

all_features_test.values.tolist()

#print(map(lambda x,y : x == y , all_features_test.values.tolist() , valid_col))

#test_pred = model_gb.predict(all_features_test)

#
# test_pred = np.expm1(test_pred)
# print("Root Mean Squared Error")
# print(sqrt(mean_squared_error(y_train_values, y_pred)))
#
# test_pred = test_pred.reshape(-1,1)
# test_pred = pd.DataFrame.from_records(test_pred)
#
#
# submit = pd.concat([test['Id'],test_pred], axis=1)
# submit.columns = ["Id", "SalePrice"]
#
# submit.to_csv("../data/submission.csv", mode='w', index = False)





