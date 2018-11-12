import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
import xgboost as xgb
from scipy.special import boxcox1p, boxcox


"""
load data
"""
train_data = pd.read_csv('../Basant/p_train.csv')
test_data = pd.read_csv('../Basant/p_test.csv')
y_train_values = pd.read_csv('../Basant/actual_price.csv')

train_set_id = train_data['Id']
test_set_id = test_data['Id']
y_train_values = y_train_values['log_SalePrice'].values

train_data['data_type'] = 0
test_data['data_type'] = 1

combined_data = pd.concat((train_data, test_data))



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

"""  split """

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

print(sale_price_lasso)

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
print('xgboost')

gbm = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)\
    .fit(train_data, y_train_values,
         early_stopping_rounds=5,
         eval_set=[(train_data, y_train_values)], verbose=False)
predictions = gbm.predict(test_data)
sale_price_xgb = np.expm1(gbm.predict(predict_data))

submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_xgb
})
submission.to_csv('submission_xgboost_take05.csv', index=False)

print("Xgboost Root Mean Squared Error")
print(sqrt(mean_squared_error(y_train_values, gbm.predict(train_data))))


sale_price_ensemble = (sale_price_enet + sale_price_lasso + sale_price_xgb)/3

submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_ensemble
})
submission.to_csv('submission_ensemble_take05.csv', index=False)

