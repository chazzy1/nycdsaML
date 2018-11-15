import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
import lightgbm as lgb
from utils.utils import *

"""
load data
"""
train_set = pd.read_csv('../data/train.csv')
test_set = pd.read_csv('../data/test.csv')

"""
Remove Outliers
"""
train_set.drop(train_set[(train_set['GrLivArea'] > 4000) & (train_set['SalePrice'] < 300000)].index, inplace=True)
train_set.drop(train_set[(train_set['OverallQual']<5) & (train_set['SalePrice']>200000)].index, inplace=True)
train_set.drop(train_set[(train_set['GrLivArea']>4000) & (train_set['SalePrice']<300000)].index, inplace=True)
train_set.reset_index(drop=True, inplace=True)


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
Drop some features ( by Missing Ratio )
"""
## find missing data
total = combined_data.isnull().sum().sort_values(ascending=False)
percent = (combined_data.isnull().sum()/combined_data.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

# drop columns over 40% what fill with NAN
features_to_drop = missing_data[missing_data['Percent'] > 0.4].index
features_to_drop = features_to_drop.union(['GarageArea'])
combined_data.drop(features_to_drop, axis=1, inplace=True)

## ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
## ## GarageCars & GagageArea has Multicollinearity so take 1 feature
## So remove 'GarageArea'

"""About size of house """
#
# combined_data['TotalSF'] = combined_data['1stFlrSF'] + combined_data['2ndFlrSF']
# combined_data.drop(['TotalBsmtSF','1stFlrSF','2ndFlrSF'], axis=1, inplace=True)
combined_data.drop(['1stFlrSF','2ndFlrSF'], axis=1, inplace=True)

"""
fix NaN
"""

categorical_with_nan = get_columns_with_nan(combined_data[get_categorical_columns(combined_data)])

for col in categorical_with_nan:
    combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])

numerical_with_nan = get_columns_with_nan(combined_data[get_numeric_columns(combined_data)])

for col in numerical_with_nan:
    combined_data[col] = combined_data[col].fillna(0)

"""
encode categorical features
"""
from sklearn.preprocessing import LabelEncoder

cols_encoding_needed = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional',   'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond',
        'YrSold', 'MoSold')

# cols_encoding_needed = ('BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
#         'ExterQual', 'ExterCond','HeatingQC', 'KitchenQual', 'BsmtFinType1',
#         'BsmtFinType2', 'Functional',   'BsmtExposure', 'GarageFinish', 'LandSlope',
#         'LotShape', 'PavedDrive', 'Street', 'CentralAir', 'MSSubClass', 'OverallCond')

#
# cols_encoding_needed = combined_data[get_categorical_columns(combined_data)]
#
# for col in cols_encoding_needed:
#     lbl = LabelEncoder()
#     lbl.fit(list(combined_data[col].values))
#     combined_data[col] = lbl.transform(list(combined_data[col].values))
#

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


"""
elasticNet fit
"""
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.0005, l1_ratio=0.9)

model_enet = enet.fit(train_data, y_train_values)

print("ElasticNet Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, model_enet.predict(train_data))))

sale_price_enet = np.expm1(model_enet.predict(predict_data))

"""
xgboost
"""
xgbm = xgb.XGBRegressor(
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
xgbm.fit(train_data, y_train_values,
         early_stopping_rounds=5,
         eval_set=[(train_data, y_train_values)], verbose=False)

predictions = xgbm.predict(predict_data)
sale_price_xgb = np.expm1(xgbm.predict(predict_data))

print("Xgboost Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, xgbm.predict(train_data))))

"""
LightGBM
"""
cat_features = []
for i, c in enumerate(train_data.columns):
    if('_cat' in c):
         cat_features.append(c)

lgb_model = lgb.LGBMRegressor(objective='regression',num_leaves=40,
                              learning_rate=0.01, n_estimators=4000,
                              bagging_fraction = 0.6,
                              bagging_freq = 6, feature_fraction = 0.6,
                              feature_fraction_seed=9, bagging_seed=42,
                              seed=42, metric = 'rmse' ,verbosity=-1,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
lgb_model.fit(train_data, y_train_values, early_stopping_rounds=5,
          eval_set=[(train_data, y_train_values)], verbose=False)

sale_price_lgb = np.expm1(lgb_model.predict(predict_data))

print("lightGBM Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, lgb_model.predict(train_data))))

#
# submission_xg = pd.DataFrame({
#     "Id": test_set_id,
#     "SalePrice": sale_price_xgb
# })
# submission_xg.to_csv('submission_xgboost.csv', index=False)
#
# """ print out ansemble """
# sale_price_ensemble = ( sale_price_enet  + sale_price_lasso + sale_price_xgb )/3
#
# submission = pd.DataFrame({
#     "Id": test_set_id,
#     "SalePrice": sale_price_ensemble
# })
# submission.to_csv('submission_ensemble.csv', index=False)

"""" Ensemble Weights """
from scipy.optimize import minimize
clfs = [lasso,enet,xgbm,lgb_model]

predictions = []
for clf in clfs:
    predictions.append(clf.predict(train_data)) # listing all our predictions

def mse_func(weights):
    # scipy minimize will pass the weights as a numpy array
    final_prediction = 0
    for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
    return mean_squared_error(y_train_values, final_prediction)

starting_values = [0.5]*len(predictions) # minimize need a starting value
bounds = [(0,1)]*len(predictions) # weights are bound between 0 and 1
res = minimize(mse_func,
               starting_values,
               bounds = bounds,
               method='SLSQP'
              )
print('Result Assessment: {message_algo}'.format(message_algo = res['message']))
print('Ensemble Score: {best_score}'.format(best_score = res['fun']))
print('Best Weights: {weights}'.format(weights = res['x']))

## lasso,enet,xgbm,lgb_model]
sale_price_ensemble = ( sale_price_lasso*res['x'][0] +
          sale_price_enet*res['x'][1] +
          sale_price_xgb * res['x'][2] +
          sale_price_lgb * res['x'][3] )

submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_ensemble
})
submission.to_csv('submission_ensemble_all.csv', index=False)
