import sys
sys.path.append('../')
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet

from scipy.stats import zscore

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from mlxtend.regressor import StackingRegressor
from utils.transform import *
pd.options.mode.chained_assignment = None
from sklearn.svm import SVR
import xgboost as xgb
import lightgbm as lgb


def main():
    """
    load data
    """
    train_set = pd.read_csv('./data/train.csv')
    test_set = pd.read_csv('./data/test.csv')

    """
    Remove Outliers
    """
    z = np.abs(zscore(train_set[get_numeric_columns(train_set)]))
    row, col = np.where(z > 4)
    df = pd.DataFrame({"row": row, "col": col})
    rows_count = df.groupby(['row']).count()

    outliers = rows_count[rows_count.col > 2].index
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
        ('OutlierRemover', OutlierRemover()),
        ('NaNImputer', NaNImputer()),
        ('NaNRemover', NaNRemover()),
        ('AdditionalFeatureGenerator', AdditionalFeatureGenerator()),
        ('TypeTransformer', TypeTransformer()),
        ('ErrorImputer', ErrorImputer()),
        ('SkewFixer', SkewFixer()),
        #('OrdinalConverter', OrdinalConverter()),
        ('Scaler', Scaler()),
        #('FeatureDropper', FeatureDropper()),
        ('Dummyfier', Dummyfier()),
        ('FeatureDropper2', FeatureDropper2()),
    ])


    transformed_data = transform_pipeline.transform(combined_data)
    train_data = transformed_data[:train_set_rows]
    predict_data = transformed_data[train_set_rows:]

    transformed_data.to_csv('transformed_Data.csv', index=False)

    """
    try various regressors
    """

    rf_param = {
        'max_depth': [3, 4, 5],
        'min_samples_leaf': [3, 4, 5],
        'n_estimators': [5, 7, 10]
    }
    lso_param = {'alpha': [0.0003, 0.0004, 0.0005,
                          0.0006, 0.0007, 0.0008],
                'max_iter': [10000], "normalize": [False]}

    lso_param = {'alpha': [0.0005],
                 "normalize": [False]}


    elnet_param = {'alpha': [0.0003, 0.0004, 0.0005],
                   'l1_ratio': [0.9, 0.95, 0.99, 1],
                   'max_iter': [10000]}

    elnet_param = {'alpha': [0.0002, 0.0003],
                   'l1_ratio': [0.8, 0.9],
                   'max_iter': [10000]}

    ridge_param = {'alpha': [10, 10.1, 10.2, 10.3, 10.4, 10.5]}
    ridge_param = {'alpha': [10.5, 10.6]}
    svr_param = {'gamma': [1e-08, 1e-09],
                 'C': [100000, 110000],
                 'epsilon': [1, 0.1, 0.01]
                 }
    svr_param = {'gamma': [1e-08],
                 'C': [100000],
                 'epsilon': [0.1]
                 }

    xgb_param = {'learning_rate': [0.01],
                 'n_estimators': [3460],
                 'max_depth': [3],
                 'min_child_weight': [0],
                 'gamma': [0],
                 'subsample': [0.7],
                 'colsample_bytree': [0.7],
                 'objective': ['reg:linear'],
                 'scale_pos_weight': [1],
                 'seed': [27],
                 'reg_alpha': [0.00006]
                 }

    lgb_params = {
        'objective': ['regression'],
        'num_leaves': [255],
        'max_depth': [8],
        'bagging_seed': [3],
        #'boosting_type': ['gbdt'],
        #'min_sum_hessian_in_leaf': [100],
        #'learning_rate': np.linspace(0.05, 0.1, 2),
        #'bagging_fraction': np.linspace(0.7, 0.9, 2),
        #'bagging_freq': np.linspace(30, 50, 3, dtype='int'),
        #'max_bin': [15, 63],
    }


    #rf = get_best_estimator(train_data, y_train_values, estimator=RandomForestRegressor(),
    #                        params=rf_param, n_jobs=4)
    elnet = get_best_estimator(train_data, y_train_values, estimator=ElasticNet(),
                               params=elnet_param, n_jobs=4)

    lso = get_best_estimator(train_data, y_train_values, estimator=Lasso(),
                             params=lso_param, n_jobs=4)

    rdg = get_best_estimator(train_data, y_train_values, estimator=Ridge(),
                             params=ridge_param, n_jobs=4)
    svr = get_best_estimator(train_data, y_train_values, estimator=SVR(),
                             params=svr_param, n_jobs=4)
    xgbo = get_best_estimator(train_data, y_train_values, estimator=xgb.XGBRegressor(),
                             params=xgb_param, n_jobs=4)

    lbm = get_best_estimator(train_data, y_train_values, estimator=lgb.LGBMRegressor(),
                             params=lgb_params)


    model = StackingRegressor(
        regressors=[elnet, lso, rdg, svr, xgbo, lbm],
        meta_regressor=SVR(kernel='rbf'),
        #meta_regressor=Lasso(alpha=0.0001)
    )

    model.fit(train_data, y_train_values)

    stacked = model.predict(predict_data)
    """
    11446
    ensembled = np.expm1((0.2 * elnet.predict(predict_data)) +
                         (0.2 * lso.predict(predict_data)) +
                         (0.1 * rdg.predict(predict_data)) +
                         (0.2 * xgbo.predict(predict_data)) +
                         (0.3 * stacked))
    """

    """
    11435
    ensembled = np.expm1((0.2 * elnet.predict(predict_data)) +
                         (0.2 * lso.predict(predict_data)) +
                         (0.1 * rdg.predict(predict_data)) +
                         (0.3 * xgbo.predict(predict_data)) +
                         (0.2 * stacked))
    """

    """
    11431
    """
    ensembled = np.expm1((0.2 * elnet.predict(predict_data)) +
                         (0.2 * lso.predict(predict_data)) +
                         (0.1 * rdg.predict(predict_data)) +
                         (0.2 * xgbo.predict(predict_data)) +
                         (0.1 * lbm.predict(predict_data)) +
                         (0.2 * stacked))







    df = pd.DataFrame({"price": ensembled})
    q1 = df["price"].quantile(0.0042)
    q2 = df["price"].quantile(0.99)

    df["price"] = df["price"].apply(lambda x: x if x > q1 else x * 0.77)
    df["price"] = df["price"].apply(lambda x: x if x < q2 else x * 1.1)




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
