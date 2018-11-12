import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
from scipy.special import boxcox1p, boxcox

"""
load data
"""
train_set = pd.read_csv('../data/train.csv')
test_set = pd.read_csv('../data/test.csv')



"""
Remove Outliers
"""
outliers = train_set[(train_set['GrLivArea'] > 4000) & (train_set['SalePrice'] < 300000)].index
train_set.drop(outliers, inplace=True)


"""
fix salePrice skewness
"""
train_set["SalePrice"] = np.log1p(train_set["SalePrice"])
y_train_values = train_set["SalePrice"].values



"""
prepare combined data.
We need to concat train_set and test_set to apply EDA consistently.
but in training phase, we need to separate train_set from combined data.
so, before concatenation, create data_type column and assign 0 to train_set and 1 to test_Set
"""
train_set_id = train_set['Id']
test_set_id = test_set['Id']

train_set.drop('Id', axis=1, inplace=True)
test_set.drop('Id', axis=1, inplace=True)

train_set.drop('SalePrice', axis=1, inplace=True)

train_set['data_type'] = 0
test_set['data_type'] = 1

combined_data = pd.concat((train_set, test_set))

"""
Drop some features
"""
#missing data
total = combined_data.isnull().sum().sort_values(ascending=False)
percent = (combined_data.isnull().sum()/combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

features_to_drop = missing_data[missing_data['Percent'] > 0.4].index
features_to_drop = features_to_drop.union(['GarageArea'])

combined_data.drop(features_to_drop, axis=1, inplace=True)
## ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
## ## GarageCars & GagageArea has Multicollinearity so take 1 feature
## So remove 'GarageArea'

"""sum of ch"""


"""
fix NaN
"""
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType',
            'MSZoning', 'Functional', 'Electrical','KitchenQual', 'Exterior1st',
            'Exterior2nd', 'SaleType', 'MSSubClass'
            ]:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])


for col in ['GarageYrBlt', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
            'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea',
            'Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType',
            'LotFrontage'
            ]:
    combined_data[col] = combined_data[col].fillna(0)

"""
encode categorical features
"""
from sklearn.preprocessing import LabelEncoder
cols_encoding_needed = ( 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional',   'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')


for col in cols_encoding_needed:
    lbl = LabelEncoder()
    lbl.fit(list(combined_data[col].values))
    combined_data[col] = lbl.transform(list(combined_data[col].values))


"""
fix numeric features skewness by applying boxcox
"""
numeric_columns = combined_data.dtypes[combined_data.dtypes != "object"].index
numeric_columns = numeric_columns.drop("data_type")


from scipy.stats import skew
skewed_columns = combined_data[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewed_columns.index

for feat in skewed_features:
    combined_data[feat] = boxcox1p(combined_data[feat], 0.15)



"""
scaler
"""
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
combined_data[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])


"""
get dummies
"""

combined_data = pd.get_dummies(combined_data)

train_data = combined_data[combined_data['data_type'] == 0].drop('data_type', axis=1)
predict_data = combined_data[combined_data['data_type'] == 1].drop('data_type', axis=1)

"""
lasso fit
"""
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.set_params(alpha=0.0005, normalize=True)
model_lasso = lasso.fit(train_data, y_train_values)


print("Lasso Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, model_lasso.predict(train_data))))


sale_price_lasso = np.expm1(model_lasso.predict(predict_data))


"""drop columns from lasso """
#
# model_coef = model_lasso.coef_.tolist()
# model_matrix = np.concatenate((train_data.columns.values.reshape(-1,1),
#                                model_lasso.coef_.reshape(-1,1)), axis=1)
# model_matrix = pd.DataFrame.from_records(model_matrix)
# model_matrix.columns = ['feature','coef']
#
# model_matrix = model_matrix[model_matrix['coef'] != 0 ]
# valid_col = model_matrix['feature'].values
#
# train_data_new   = train_data.loc[:, train_data.columns.isin(valid_col)]
# predict_data_new = predict_data.loc[:, predict_data.columns.isin(valid_col)]
#
# print(train_data['MSSubClass'].head(5))
# print(predict_data.info())



"""
elasticNet fit
"""
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.0005, l1_ratio=0.9)

model_enet = enet.fit(train_data, y_train_values)

print("ElasticNet Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, model_enet.predict(train_data))))

sale_price_enet = np.expm1(model_enet.predict(predict_data))

sale_price_ensemble = (sale_price_enet + sale_price_lasso)/2

"""
xgboost
"""
print('xgboost')


gbm = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)
gbm.fit(train_data, y_train_values,
         early_stopping_rounds=5,
         eval_set=[(train_data, y_train_values)], verbose=False)

# gbm = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)\
#     .fit(train_data, y_train_values,
#          early_stopping_rounds=5,
#          eval_set=[(train_data, y_train_values)], verbose=False)
predictions = gbm.predict(predict_data)
sale_price_xgb = np.expm1(gbm.predict(predict_data))

submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_xgb
})
submission.to_csv('submission_xgboost.csv', index=False)

print("Xgboost Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, gbm.predict(train_data))))


sale_price_ensemble = ( sale_price_enet  + sale_price_lasso + sale_price_xgb )/3

submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_ensemble
})
submission.to_csv('submission_ensemble.csv', index=False)

