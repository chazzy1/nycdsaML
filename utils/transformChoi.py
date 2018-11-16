import sys
sys.path.append('../')

import pandas as pd
import numpy as np
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

        #Fix wrong data

        pIdx = X[(X['PoolQC'].isnull()) & (X['PoolArea'] > 0)].index
        X['PoolQC'].loc[pIdx] = X['PoolQC'].mode()[0]
        X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

        gIdx = X[X['GarageYrBlt'] == 2207].index
        X['GarageYrBlt'].loc[gIdx] = 2007

        # X["LotAreaCut"] = pd.qcut(X.LotArea, 10)
        # X['LotFrontage'] = X.groupby(['LotAreaCut', 'Neighborhood'])['LotFrontage'].transform(
        #     lambda x: x.fillna(x.median()))
        #
        # X.drop('LotAreaCut', axis=1, inplace=True)
        categorical_with_nan = get_columns_with_nan(X[get_categorical_columns(X)])

        categorical_with_nan.remove('PoolQC')

        for col in categorical_with_nan:
            X[col].fillna(X[col].mode()[0], inplace=True)

        numerical_with_nan = get_columns_with_nan(X[get_numeric_columns(X)])

        print(numerical_with_nan)
        for col in numerical_with_nan:
            X[col].fillna(X[col].median(), inplace=True)


        # X['TotalArea'] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"] + X["GarageArea"]
        # X['GrLivArea_OverallQual'] = X['GrLivArea'] * X['OverallQual']
        #
        # X["TotalHouse"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]
        # X["Rooms"] = X["FullBath"] + X["TotRmsAbvGrd"]
        # X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
        # X['TotBathrooms']  = X['FullBath']  + (X['HalfBath'] * 0.5) + X['BsmtFullBath'] + (X['BsmtHalfBath'] * 0.5)
        #
        # X['isNew'] = np.where(X['YrSold'] == X['YearBuilt'], 1, 0)
        #
        # X["MoSold"] = X["MoSold"].astype(str)
        # X["YrSold"] = X["YrSold"].astype(str)
        # X["isNew"] = X["isNew"].astype(str)
        #
        # X["MSSubClass"] = X["MSSubClass"].astype(str)
        # X["OverallCond"] = X["OverallCond"].astype(str)

        # X['isCentralAir'] = np.where(X['CentralAir'] == 'Y', 1, 0)
        # X['Age'] = X['YrSold'] - X['YearRemodAdd']

        #X["isCentralAir"] = X["isCentralAir"].astype(str)


        #X['isRemod'] = np.where(X['YearRemodAdd'] == X['YearBuilt'], '0', '1') #0=No Remodeling, 1=Remodeling
        #X['isNew'] = np.where(X['YrSold'] == X['YearBuilt'], 1, 0)

        ## Legacy
        # X["PorchArea"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        # X["TotalPlace"] = X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]  + X["OpenPorchSF"] \
        #                   + X[
        #     "EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]

        # X["Bsmt"] = X["BsmtFinSF1"] + X["BsmtFinSF2"] + X["BsmtUnfSF"]
        # X["+_TotalHouse_OverallQual"] = X["TotalHouse"] * X["OverallQual"]
        # X["+_GrLivArea_OverallQual"] = X["GrLivArea"] * X["OverallQual"]
        # X["+_oMSZoning_TotalHouse"] = X["oMSZoning"] * X["TotalHouse"]
        # X["+_oMSZoning_OverallQual"] = X["oMSZoning"] + X["OverallQual"]

        #
        # NumStr = ["MSSubClass", "BsmtFullBath", "BsmtHalfBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "MoSold",
        #           "YrSold", "YearBuilt", "YearRemodAdd", "LowQualFinSF", "GarageYrBlt"]
        #
        # for col in NumStr:
        #     X[col] = X[col].astype(str)
        #
        # X["oMSSubClass"] = X.MSSubClass.map({'180': 1,
        #                                      '30': 2, '45': 2,
        #                                      '190': 3, '50': 3, '90': 3,
        #                                      '85': 4, '40': 4, '160': 4,
        #                                      '70': 5, '20': 5, '75': 5, '80': 5, '150': 5,
        #                                      '120': 6, '60': 6})

        # X["MSZoning"] = X.MSZoning.map({'C (all)': 1, 'RH': 2, 'RM': 2, 'RL': 3, 'FV': 4})
        #
        # X["oNeighborhood"] = X.Neighborhood.map({'MeadowV': 1,
        #                                          'IDOTRR': 2, 'BrDale': 2,
        #                                          'OldTown': 3, 'Edwards': 3, 'BrkSide': 3,
        #                                          'Sawyer': 4, 'Blueste': 4, 'SWISU': 4, 'NAmes': 4,
        #                                          'NPkVill': 5, 'Mitchel': 5,
        #                                          'SawyerW': 6, 'Gilbert': 6, 'NWAmes': 6,
        #                                          'Blmngtn': 7, 'CollgCr': 7, 'ClearCr': 7, 'Crawfor': 7,
        #                                          'Veenker': 8, 'Somerst': 8, 'Timber': 8,
        #                                          'StoneBr': 9,
        #                                          'NoRidge': 10, 'NridgHt': 10})
        #
        # X["Condition1"] = X.Condition1.map({'Artery': 1,
        #                                      'Feedr': 2, 'RRAe': 2,
        #                                      'Norm': 3, 'RRAn': 3,
        #                                      'PosN': 4, 'RRNe': 4,
        #                                      'PosA': 5, 'RRNn': 5})
        #
        # X["BldgType"] = X.BldgType.map({'2fmCon': 1, 'Duplex': 1, 'Twnhs': 1, '1Fam': 2, 'TwnhsE': 2})
        #
        # X["HouseStyle"] = X.HouseStyle.map({'1.5Unf': 1,
        #                                      '1.5Fin': 2, '2.5Unf': 2, 'SFoyer': 2,
        #                                      '1Story': 3, 'SLvl': 3,
        #                                      '2Story': 4, '2.5Fin': 4})
        #
        # X["Exterior1st"] = X.Exterior1st.map({'BrkComm': 1,
        #                                        'AsphShn': 2, 'CBlock': 2, 'AsbShng': 2,
        #                                        'WdShing': 3, 'Wd Sdng': 3, 'MetalSd': 3, 'Stucco': 3, 'HdBoard': 3,
        #                                        'BrkFace': 4, 'Plywood': 4,
        #                                        'VinylSd': 5,
        #                                        'CemntBd': 6,
        #                                        'Stone': 7, 'ImStucc': 7})
        #
        # X["MasVnrType"] = X.MasVnrType.map({'BrkCmn': 1, 'None': 1, 'BrkFace': 2, 'Stone': 3})
        #
        # X["ExterQual"] = X.ExterQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
        #
        # X["Foundation"] = X.Foundation.map({'Slab': 1,
        #                                      'BrkTil': 2, 'CBlock': 2, 'Stone': 2,
        #                                      'Wood': 3, 'PConc': 4})
        #
        # X["BsmtQual"] = X.BsmtQual.map({'Fa': 2, 'None': 1, 'TA': 3, 'Gd': 4, 'Ex': 5})
        #
        # X["BsmtExposure"] = X.BsmtExposure.map({'None': 1, 'No': 2, 'Av': 3, 'Mn': 3, 'Gd': 4})
        #
        # X["Heating"] = X.Heating.map({'Floor': 1, 'Grav': 1, 'Wall': 2, 'OthW': 3, 'GasW': 4, 'GasA': 5})
        #
        # X["HeatingQC"] = X.HeatingQC.map({'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
        #
        # X["KitchenQual"] = X.KitchenQual.map({'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})
        #
        # X["Functional"] = X.Functional.map({'Maj2': 1, 'Maj1': 2, 'Min1': 2, 'Min2': 2, 'Mod': 2, 'Sev': 2, 'Typ': 3})
        #
        #
        # X["GarageType"] = X.GarageType.map({'CarPort': 1, 'None': 1,
        #                                      'Detchd': 2,
        #                                      '2Types': 3, 'Basment': 3,
        #                                      'Attchd': 4, 'BuiltIn': 5})
        #
        # X["GarageFinish"] = X.GarageFinish.map({'None': 1, 'Unf': 2, 'RFn': 3, 'Fin': 4})
        #
        # X["PavedDrive"] = X.PavedDrive.map({'N': 1, 'P': 2, 'Y': 3})
        #
        # X["SaleType"] = X.SaleType.map({'COD': 1, 'ConLD': 1, 'ConLI': 1, 'ConLw': 1, 'Oth': 1, 'WD': 1,
        #                                  'CWD': 2, 'Con': 3, 'New': 3})
        #
        # X["SaleCondition"] = X.SaleCondition.map(
        #     {'AdjLand': 1, 'Abnorml': 2, 'Alloca': 2, 'Family': 2, 'Normal': 3, 'Partial': 4})

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

        total = X.isnull().sum().sort_values(ascending=False)
        percent = (X.isnull().sum() / X.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

        # drop columns over 40% what fill with NAN
        #features_to_drop = missing_data[missing_data['Percent'] > 0.4].index
        #features_to_drop = features_to_drop.union(['GarageArea'])
        #features_to_drop = (['Utilities'])
        # X.drop(features_to_drop, axis=1, inplace=True)
        X.drop(['Utilities', 'Street'], axis=1)

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