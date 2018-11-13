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

pd.options.mode.chained_assignment = None


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
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

        return X


class FeatureDropper(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, columns=[]):
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


def get_best_estimator(train_data, y_train_values, estimator=None, params={}, cv=5):
    name = estimator.__class__.__name__
    pipeline = Pipeline(steps=[
        (name, estimator),
    ])

    params = [
        {
            name+"__"+k: v for k, v in params.items()
        }
    ]

    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    grid_search = GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=cv, verbose=1)
    grid_search.fit(train_data, y_train_values)

    cvres = grid_search.cv_results_

    cvres = sorted([(sqrt(-score), para) for score, para in zip(cvres['mean_test_score'], cvres['params'])], reverse=False)
    print("best "+name)
    print(cvres)
    return grid_search.best_estimator_




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
    create data transform pipeline
    """
    transform_pipeline = Pipeline(steps=[
        ('NaNFixer', NaNFixer()),
        ('SkewFixer', SkewFixer()),
        ('Scaler', Scaler()),
        ('FeatureDropper', FeatureDropper()),
        ('Dummyfier', Dummyfier()),
        #('TrainDataSeparator', TrainDataSeparator(train_set_rows=train_set_rows)),
    ])

    transformed_data = transform_pipeline.transform(combined_data)
    train_data = transformed_data[:train_set_rows]
    predict_data = transformed_data[train_set_rows:]

    """
    try various regressors
    """

    rf = RandomForestRegressor(
        n_estimators=12,
        max_depth=3,
        n_jobs=-1
    )

    gb = GradientBoostingRegressor(
        n_estimators=40,
        max_depth=2
    )

    nn = MLPRegressor(
        hidden_layer_sizes=(90, 90),
        alpha=2.75
    )

    lso = Lasso()
    rf = get_best_estimator(train_data, y_train_values, estimator=RandomForestRegressor(),
                            params={"n_estimators": [50, 100], "max_depth": [3]})
    lso = get_best_estimator(train_data, y_train_values, estimator=Lasso(), params={"alpha": [0.0005, 0.0006], "normalize": [True, False]})

    gbm = get_best_estimator(train_data, y_train_values, estimator=xgb.XGBRegressor(),
                                params={"n_estimators": [1000], "learning_rate": [0.05, 0.01]}
                             )

    model = StackingRegressor(
        regressors=[rf, gb, nn, lso, xgb],
        meta_regressor=Lasso(alpha=0.0005)
    )

    # Fit the model on our data
    model.fit(train_data, y_train_values)

    y_pred = model.predict(train_data)
    print(sqrt(mean_squared_error(y_train_values, y_pred)))

    # Predict test set
    ensembled = np.expm1(model.predict(predict_data))

    """
    export submission data
    """
    submission = pd.DataFrame({
        "Id": test_set_id,
        "SalePrice": ensembled
    })
    submission.to_csv('submission.csv', index=False)







if __name__== "__main__":
    main()
