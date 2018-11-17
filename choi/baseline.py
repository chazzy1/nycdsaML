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

    def cv_rmse(model):
        kfolds = KFold(n_splits=5, shuffle=False, random_state=42)
        rmse = np.sqrt(-cross_val_score(model, train_data, y_train_values,
                                        scoring="neg_mean_squared_error",
                                        cv=kfolds))
        return (rmse)

    print("Randomforest  model rmse : ", cv_rmse(rf).mean())
    print("elastic model rmse : ", cv_rmse(elnet).mean())
    print("lasso model rmse : ", cv_rmse(lso).mean())
    print("ridge model rmse : ", cv_rmse(rdg).mean())
    print("svr model rmse : ", cv_rmse(svr).mean())

    model = StackingRegressor(
        regressors=[rf, elnet, lso, rdg, svr],
        meta_regressor=Lasso(alpha=0.0005)
        # meta_regressor=SVR(kernel='rbf')
    )

    # Fit the model on our data
    model.fit(train_data, y_train_values)
    print("StackingRegressor model rmse : ", cv_rmse(model).mean())

    # y_pred = model.predict(train_data)
    # print(sqrt(mean_squared_error(y_train_values, y_pred)))

    # Predict test set
    ensembled = np.expm1(model.predict(predict_data))

    # sns.scatterplot(np.expm1(rf.predict(train_data),np.expm1(y_train_values)))
    # plt.show()
    # ensembled = np.expm1(rf.predict(predict_data))

    """
    export submission data
    """
    submission = pd.DataFrame({
        "Id": test_set_id,
        "SalePrice": ensembled
    })
    submission.to_csv('submission_jiwon.csv', index=False)

if __name__== "__main__":
    main()
