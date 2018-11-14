import sys
sys.path.append('../')

import pandas as pd
from math import sqrt
from utils.utils import *
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from scipy.stats import skew
from scipy.special import boxcox1p


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


def get_best_estimator(train_data, y_train_values, estimator=None, params={}, cv=5, n_jobs=-1):
    name = estimator.__class__.__name__
    pipeline = Pipeline(steps=[
        (name, estimator),
    ])

    params = [
        {
            name+"__"+k: v for k, v in params.items()
        }
    ]
    from sklearn.model_selection import cross_val_score, KFold
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    kf = KFold(cv, shuffle=True, random_state=42).get_n_splits(train_data)
    grid_search = GridSearchCV(pipeline, param_grid=params, scoring=scorer, cv=kf, verbose=1, n_jobs=n_jobs)
    grid_search.fit(train_data, y_train_values)

    print("Estimator: {} score: ({}) best params: {}".format(name, sqrt(-grid_search.best_score_), grid_search.best_params_))
    print(grid_search.best_estimator_)
    return grid_search.best_estimator_