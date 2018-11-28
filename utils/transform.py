import sys
sys.path.append('../')

import pandas as pd
from math import sqrt
from utils.utils import *
from sklearn.pipeline import Pipeline, TransformerMixin
#from sklearn.bxgboostase import TransformerMixin

from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

class OutlierRemover(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # not implemented for now.
        # will be taken care of by individual main code.
        return X

class NaNImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[(X['PoolArea'] > 0) & (X['PoolQC'].isnull()), 'PoolQC'] = 'TA'

        X['PoolQC'].fillna('NA', inplace=True)
        X['MiscFeature'].fillna('NA', inplace=True)
        X['Alley'].fillna('NA', inplace=True)
        X['Fence'].fillna('NA', inplace=True)
        X['FireplaceQu'].fillna('NA', inplace=True)

        #X['LotFrontage'].fillna(0, inplace=True) # NaNRemover will take care of this.

        X.loc[X['GarageType'].isnull(), ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt', 'GarageType',
                                         'GarageCars', 'GarageArea']] \
            = ['NA', 'NA', 'NA', 0, 'NA', 0, 0]

        X.loc[(X['GarageCond'].isnull()) & (X['OverallCond'] == 8),
              ['GarageFinish', 'GarageQual', 'GarageCond', 'GarageYrBlt']] \
            = [
            X[X['OverallCond'] == 8]['GarageFinish'].mode()[0],
            X[X['OverallCond'] == 8]['GarageQual'].mode()[0],
            X[X['OverallCond'] == 8]['GarageCond'].mode()[0],
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

        X['BsmtHalfBath'].fillna('0', inplace=True)

        X['MasVnrType'].fillna('NA', inplace=True)

        X.loc[(X['MSZoning'].isnull()) & (X['MSSubClass'] == 20), 'MSZoning'] = 'RL'
        X.loc[(X['MSZoning'].isnull()) & (X['MSSubClass'] == 30), 'MSZoning'] = 'RM'
        X.loc[(X['MSZoning'].isnull()) & (X['MSSubClass'] == 70), 'MSZoning'] = 'RM'

        X['Functional'].fillna('Typ', inplace=True)

        X['KitchenQual'].fillna('TA', inplace=True)

        return X



class NaNRemover(TransformerMixin):
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


class AdditionalFeatureGenerator(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['TotalLivingSpace'] = (X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['1stFlrSF'] + X['2ndFlrSF'])
        X['Total_Bathrooms'] = (X['BsmtFullBath'].apply(lambda x: float(x))) +\
                               (X['BsmtHalfBath'].apply(lambda x: float(x)*0.4)) +\
                               (X['FullBath'].apply(lambda x: float(x))) + \
                               (X['HalfBath'].apply(lambda x: float(x)*0.4))

        X['has2ndfloor'] = X['2ndFlrSF'].apply(lambda x: 'Y' if x > 0 else 'N')

        X['hasbsmt'] = X['TotalBsmtSF'].apply(lambda x: 'Y' if x > 0 else 'N')
        features_to_drop = ['BsmtFinSF1','BsmtFinSF2','1stFlrSF','2ndFlrSF','FullBath',
                            'HalfBath','BsmtFullBath','BsmtHalfBath']
        X.drop(features_to_drop, axis=1, inplace=True)
        return X


class TypeTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.update(X['MSSubClass'].astype('str'))
        #X['YrSold'] = X['YrSold'].astype(str)
        #X['MoSold'] = X['MoSold'].astype(str)
        return X

class ErrorImputer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.loc[X['GarageYrBlt'] == 2207, 'GarageYrBlt'] = 2007

        return X


class SkewFixer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = get_numeric_columns(X)

        skewed_columns = X[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewed_columns = skewed_columns[abs(skewed_columns) > 0.5]
        skewed_features = skewed_columns.index

        for feat in skewed_features:
            #raise ValueError("Data must be positive.") occurs. 0인거 같음. 그냥 +1 ?
            X[feat] = boxcox1p(X[feat], boxcox_normmax(X[feat]+1))

        return X


class Scaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        numeric_columns = get_numeric_columns(X)
        scaler = preprocessing.StandardScaler()
        #scaler = preprocessing.RobustScaler()
        X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

        return X


class OrdinalConverter(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):

        ordinal_columns = [
            "ExterCond",
            "ExterQual",
            "BsmtQual",
            "BsmtCond",
            "BsmtExposure",
            "HeatingQC",
            "KitchenQual",
            "Functional",
            "GarageCond",

        ]

        ordinals = {
            "ExterCond": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
            "ExterQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
            "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "NA": 0},
            "HeatingQC": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
            "KitchenQual": {"Ex": 4, "Gd": 3, "TA": 2, "Fa": 1, "Po": 0},
            "Functional": {"Typ": 7, "Min1": 6, "Min2": 5, "Mod": 4, "Maj1": 3, "Maj2": 2, "Sev": 1, "Sal": 0},
            "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NA": 0},
        }

        X.replace(ordinals, inplace=True)

        return X


class FeatureDropper(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, columns=[]):
        features_to_drop = [
            'PoolQC',
        ]

        X.drop(features_to_drop, axis=1, inplace=True)

        return X


class FeatureDropper2(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, columns=[]):
        features_to_drop = [
            'MSSubClass_150',
            'MSZoning_C (all)',
            'Utilities_NoSeWa',
            'Condition2_RRAe',
            'Condition2_RRAn',
            'Condition2_RRNn',
            'RoofMatl_Membran',
            'RoofMatl_Metal',
            'RoofMatl_Roll',
            'Exterior1st_AsphShn',
            'Exterior1st_CBlock',
            'Exterior1st_Stone',
            'Exterior1st_ImStucc',
            'Exterior2nd_Other',
            'Heating_Floor',
            'Electrical_Mix',
            'MiscFeature_TenC',
        ]




        #print(X.columns)
        #X.to_csv('modified_data.csv', index=False)


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