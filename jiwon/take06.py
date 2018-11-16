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
from utils.transform import *
pd.options.mode.chained_assignment = None

from sklearn.svm import SVC
from sklearn.svm import SVR




def main():
    """
    load data
    """
    train_set = pd.read_csv('../data/train.csv')
    test_set = pd.read_csv('../data/test.csv')

    #Without outlier remover, with basic nanRemover 0.12416413124809748

    """
    Remove Outliers
    """
    outliers = train_set[ train_set['GrLivArea'] > 4500 ].index
    print(outliers)

    outliers = [197, 523, 691, 854, 1182, 1298]


    train_set.drop(outliers, inplace=True)

    #With outlier remover 0.10970218665126451

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
        ('OutlierRemover', OutlierRemover()),
        ('NaNImputer', NaNImputer()),
        ('NaNRemover', NaNRemover()),
        ('AdditionalFeatureGenerator', AdditionalFeatureGenerator()),
        ('TypeTransformer', TypeTransformer()),
        ('ErrorImputer', ErrorImputer()),
        ('SkewFixer', SkewFixer()),
        ('Scaler', Scaler()),
        ('FeatureDropper', FeatureDropper()),
        ('Dummyfier', Dummyfier()),
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





    svr = get_best_estimator(train_data, y_train_values, estimator=Lasso(),
                             params={
                                 'alpha': [0.00051, 0.0005, 0.00049, 0.00048, 0.00047],
                             },
                            n_jobs=4)
    model=svr

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
