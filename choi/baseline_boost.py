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
from sklearn.model_selection import KFold, cross_val_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
import xgboost as xgb
import lightgbm as lgb
from utils.transform import *
pd.options.mode.chained_assignment = None

from sklearn.svm import SVC
from sklearn.svm import SVR

import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


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

    rf_param = {
        # 'bootstrap': [True],
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [3, 4, 5],
        'n_estimators': [5, 7, 10]
    }
    ls_param = {'alpha': [0.0003, 0.0004, 0.0005,
                          0.0006, 0.0007, 0.0008],
                'max_iter': [10000], "normalize": [False]}

    elnet_param = {'alpha': [0.0003, 0.0004, 0.0005],
                   'l1_ratio': [0.9, 0.95, 0.99, 1],
                   'max_iter': [10000]}

    ridge_param = {'alpha': [10, 10.1, 10.2, 10.3, 10.4, 10.5]}

    svr_param = {'gamma': [1e-08, 1e-09],
                 'C': [100000, 110000],
                 'epsilon': [1, 0.1, 0.01]
                 }
    gbm_param = {"n_estimators": [1000],
                 'min_child_weight': [1, 5],
                 'gamma': [0.1, 0.2],
                 'subsample': [0.6],
                 'colsample_bytree': [0.6],
                 'max_depth': [3, 4],
                 'eta': [0.01],
                 'eval_metric': ['mae']
                 }

    lgb_params = {
        'objective': ['regression'],
        'num_leaves': [255],
        'max_depth': [8],
        'bagging_seed': [3],
        'boosting_type': ['gbdt']
        ,
        'min_sum_hessian_in_leaf': [100],
        'learning_rate': np.linspace(0.05, 0.1, 2),
        'bagging_fraction': np.linspace(0.7, 0.9, 2),
        'bagging_freq': np.linspace(30, 50, 3, dtype='int'),
        'max_bin': [15, 63],
    }



    rf = get_best_estimator(train_data, y_train_values, estimator=RandomForestRegressor(),
                            params=rf_param, n_jobs=4)
    elnet = get_best_estimator(train_data, y_train_values, estimator=ElasticNet(),
                               params=elnet_param, n_jobs=4)
    lso = get_best_estimator(train_data, y_train_values, estimator=Lasso(),
                             params=ls_param, n_jobs=4)
    rdg = get_best_estimator(train_data, y_train_values, estimator=Ridge(),
                             params=ridge_param, n_jobs=4)
    svr = get_best_estimator(train_data, y_train_values, estimator=SVR(),
                             params=svr_param, n_jobs=4)


    # gbm = get_best_estimator(train_data, y_train_values, estimator=xgb.XGBRegressor(),
    #                          params=gbm_param, n_jobs=4)
    # lbm = get_best_estimator(train_data, y_train_values, estimator=lgb.LGBMRegressor(),
    #                          params=lgb_params, n_jobs=4)

    def cv_rmse(model):
        kfolds = KFold(n_splits=5, shuffle=True, random_state=42)
        rmse = np.sqrt(-cross_val_score(model, train_data, y_train_values,
                                        scoring="neg_mean_squared_error",
                                        cv=kfolds))
        return (rmse)

    print("Randomforest  model rmse : ", cv_rmse(rf).mean())
    print("elastic model rmse : ", cv_rmse(elnet).mean())
    print("lasso model rmse : ", cv_rmse(lso).mean())
    print("ridge model rmse : ", cv_rmse(rdg).mean())
    print("svr model rmse : ", cv_rmse(svr).mean())
    # print("xgboost model rmse : ", cv_rmse(gbm).mean())
    # print("lightgbm model rmse : ", cv_rmse(lbm).mean())

    model = StackingRegressor(
        regressors=[rf, elnet, lso, rdg, svr ],
        meta_regressor=Lasso(alpha=0.0005)
    )

    # Fit the model on our data
    model.fit(train_data, y_train_values)
    print("StackingRegressor model rmse : ", cv_rmse(model).mean())

    # y_pred = model.predict(train_data)
    # print(sqrt(mean_squared_error(y_train_values, y_pred)))

    # Predict test set
    ensembled = np.expm1(model.predict(predict_data))

    """
    export submission data
    """
    submission = pd.DataFrame({
        "Id": test_set_id,
        "SalePrice": ensembled
    })
    submission.to_csv('submission_stack_boost.csv', index=False)

    """" Ensemble Weights """
    from scipy.optimize import minimize
    regressors = [rf, elnet, lso, rdg, svr ]

    predictions = []
    for clf in regressors:
        predictions.append(clf.predict(train_data))  # listing all our predictions

    def mse_func(weights):
        # scipy minimize will pass the weights as a numpy array
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
            final_prediction += weight * prediction
        return mean_squared_error(y_train_values, final_prediction)

    starting_values = [0.5] * len(predictions)  # minimize need a starting value
    bounds = [(0, 1)] * len(predictions)  # weights are bound between 0 and 1
    res = minimize(mse_func,
                   starting_values,
                   bounds=bounds,
                   method='SLSQP'
                   )
    print('Result Assessment: {message_algo}'.format(message_algo=res['message']))
    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))

    ##  All
    sale_price_ensemble = (np.expm1(rf.predict(predict_data)) * res['x'][0] +
                           np.expm1(elnet.predict(predict_data)) * res['x'][1] +
                           np.expm1(lso.predict(predict_data)) * res['x'][2] +
                           np.expm1(rdg.predict(predict_data)) * res['x'][3] +
                           np.expm1(svr.predict(predict_data)) * res['x'][4] +
                           np.expm1(gbm.predict(predict_data)) * res['x'][5] +
                           np.expm1(gbm.predict(predict_data)) * res['x'][6])

    submission = pd.DataFrame({
        "Id": test_set_id,
        "SalePrice": sale_price_ensemble
    })
    submission.to_csv('submission_stack_average.csv', index=False)



if __name__== "__main__":
    main()
