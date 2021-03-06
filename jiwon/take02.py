import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from math import sqrt
from utils.utils import *


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
cols_encoding_needed = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
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

#robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])


"""
Drop some features
"""

features_to_drop = []

combined_data.drop(features_to_drop, axis=1, inplace=True)








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



sale_price_ensemble = (sale_price_enet + sale_price_lasso)/2

"""
export submission data
"""
submission = pd.DataFrame({
    "Id": test_set_id,
    "SalePrice": sale_price_ensemble
})
submission.to_csv('submission.csv', index=False)


