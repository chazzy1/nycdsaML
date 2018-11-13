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

pd.options.mode.chained_assignment = None


#train_data = combined_data[combined_data['data_type'] == 0].drop('data_type', axis=1)
#predict_data = combined_data[combined_data['data_type'] == 1].drop('data_type', axis=1)
#
#
# """
# lasso fit
# """
# lasso = Lasso()
# lasso.set_params(alpha=0.0005, normalize=True)
# model_lasso = lasso.fit(train_data, y_train_values)
#
#
# print("Lasso Root Mean Squared Error")
# print(sqrt(mean_squared_error(y_train_values, model_lasso.predict(train_data))))
#
#
# sale_price_lasso = np.expm1(model_lasso.predict(predict_data))
#
# """
# elasticNet fit
# """
# from sklearn.linear_model import ElasticNet
#
# enet = ElasticNet(alpha=0.0005, l1_ratio=0.9)
#
# model_enet = enet.fit(train_data, y_train_values)
#
# print("ElasticNet Root Mean Squared Error")
# print(sqrt(mean_squared_error(y_train_values, model_enet.predict(train_data))))
#
# sale_price_enet = np.expm1(model_enet.predict(predict_data))
#
#
#
# sale_price_ensemble = (sale_price_enet + sale_price_lasso)/2
#
# """
# export submission data
# """
# submission = pd.DataFrame({
#     "Id": test_set_id,
#     "SalePrice": sale_price_ensemble
# })
# submission.to_csv('submission.csv', index=False)
#
#

class NaNFixer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_with_nan = get_columns_with_nan(X[get_categorical_columns(X)])

        for col in categorical_with_nan:
            X[col].fillna(X[col].mode()[0], inplace=True)

        numerical_with_nan = get_columns_with_nan(X[get_numeric_columns(X)])

        for col in numerical_with_nan:
            X[col].fillna(0, inplace=True)

        return X


class NaNFixer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_with_nan = get_columns_with_nan(X[get_categorical_columns(X)])

        for col in categorical_with_nan:
            X[col].fillna(X[col].mode()[0], inplace=True)

        numerical_with_nan = get_columns_with_nan(X[get_numeric_columns(X)])

        for col in numerical_with_nan:
            X[col].fillna(0, inplace=True)

        return X


class SkewFixer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = get_numeric_columns(X)

        skewed_columns = X[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]
        skewed_features = skewed_columns.index

        for feat in skewed_features:
            X[feat] = boxcox1p(X[feat], 0.15)

        return X


class Scaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = get_numeric_columns(X)
        scaler = preprocessing.StandardScaler()
        #X[numeric_columns] = scaler.fit_transform(X[numeric_columns])


        return X


class FeatureDropper(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features_to_drop = []

        X.drop(features_to_drop, axis=1, inplace=True)

        return X


class Dummyfier(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        categorical_columns = get_categorical_columns(X)

        X = pd.get_dummies(data=X, columns=categorical_columns)

        return X


class TrainDataSeparator(TransformerMixin):
    def __init__(self, train_set_rows):
        self.train_set_rows = train_set_rows

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        train_data = X[:self.train_set_rows]

        return train_data




def main():
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
    """
    train_set_id = train_set['Id']
    test_set_id = test_set['Id']

    train_set_rows = train_set.shape[0]
    test_set_rows = test_set.shape[0]

    train_set.drop('Id', axis=1, inplace=True)
    test_set.drop('Id', axis=1, inplace=True)

    train_set.drop('SalePrice', axis=1, inplace=True)

    combined_data = pd.concat((train_set, test_set))













    """
    fix NaN
    """
    # categorical_with_nan = get_columns_with_nan(combined_data[get_categorical_columns(combined_data)])
    #
    # for col in categorical_with_nan:
    #     combined_data[col] = combined_data[col].fillna(combined_data[col].mode()[0])
    #
    # numerical_with_nan = get_columns_with_nan(combined_data[get_numeric_columns(combined_data)])
    #
    # for col in numerical_with_nan:
    #     combined_data[col] = combined_data[col].fillna(0)



    """
    fix numeric features skewness by applying boxcox
    """
    # numeric_columns = combined_data.dtypes[combined_data.dtypes != "object"].index
    #
    # # numeric_columns = numeric_columns.drop("data_type")
    #
    # skewed_columns = combined_data[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    #
    # skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]
    #
    # skewed_features = skewed_columns.index
    #
    # for feat in skewed_features:
    #     combined_data[feat] = boxcox1p(combined_data[feat], 0.15)

    """
    scaler
    """
    # numeric_columns = combined_data.dtypes[combined_data.dtypes != "object"].index
    #
    # scaler = preprocessing.StandardScaler()
    # combined_data[numeric_columns] = scaler.fit_transform(combined_data[numeric_columns])

    # robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])

    """
    Drop some features
    """
    #
    # features_to_drop = []
    #
    # combined_data.drop(features_to_drop, axis=1, inplace=True)

    """
    get dummies
    """
    # categorical_columns = get_categorical_columns(combined_data)
    #
    # combined_data = pd.get_dummies(data=combined_data, columns=categorical_columns)

    train_data = combined_data[:train_set_rows]
    predict_data = combined_data[train_set_rows:]








    scorer = make_scorer(mean_squared_error, greater_is_better=False)


    pipeline = Pipeline(steps=[
        ('NaNFixer', NaNFixer()),
        ('SkewFixer', SkewFixer()),
        ('Scaler', Scaler()),
        ('FeatureDropper', FeatureDropper()),
        ('Dummyfier', Dummyfier()),
        ('TrainDataSeparator', TrainDataSeparator(train_set_rows=train_set_rows)),

        ('Lasso', Lasso()),

    ])



    params = [
        {
            'Lasso__alpha': [0.0005, 0.0001],
            'Lasso__normalize': [True, False]
        }
    ]

    grid_search = GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=5, verbose=1)
    grid_search.fit(combined_data, y_train_values)


    cvres = grid_search.cv_results_

    cvres = sorted([(sqrt(-score), para) for score, para in zip(cvres['mean_test_score'], cvres['params'])], reverse=False)
    print(cvres)



















if __name__== "__main__":
  main()
