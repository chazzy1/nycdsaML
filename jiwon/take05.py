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





def main():
    """
    load data
    """
    train_set = pd.read_csv('../data/train.csv')
    test_set = pd.read_csv('../data/test.csv')

    """
    Remove Outliers
    """
    outliers = train_set[ train_set['GrLivArea'] > 4500 ].index
    print(outliers)
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


    from sklearn.svm import SVC
    from sklearn.svm import SVR



    svr = get_best_estimator(train_data, y_train_values, estimator=SVR(),
                             params={
                                 'gamma': [1e-08, 1e-09],
                                 'C': [100000, 110000],
                                 'epsilon': [1, 0.1, 0.01]

                             },
                            n_jobs=4)
    #C = 100000, gamma = 1e-08
    model=svr

    """
    Estimator: SVR score: (0.1149444721119083) best params: {'SVR__C': 110000, 'SVR__epsilon': 0.1, 'SVR__gamma': 1e-08}
Pipeline(memory=None,
     steps=[('SVR', SVR(C=110000, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1e-08,
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))])
0.10786474107501662
    
    """

    """
    lso = Lasso()
    rf = get_best_estimator(train_data, y_train_values, estimator=RandomForestRegressor(),
                            params={"n_estimators": [50, 100], "max_depth": [3]})
    lso = get_best_estimator(train_data, y_train_values, estimator=Lasso(), params={"alpha": [0.0005, 0.0006], "normalize": [True, False]})

    gbm = get_best_estimator(train_data, y_train_values, estimator=xgb.XGBRegressor(),
                                params={"n_estimators": [1000], "learning_rate": [0.05, 0.01]}
                             )
    """


    """
    model = StackingRegressor(
        regressors=[rf, gb, nn, lso, gbm],
        meta_regressor=Lasso(alpha=0.0005)
    )

    # Fit the model on our data
    model.fit(train_data, y_train_values)
    """



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
