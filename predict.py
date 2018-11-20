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
        ('Scaler', Scaler()),

        ('Dummyfier', Dummyfier()),
        ('FeatureDropper', FeatureDropper()),
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

    model = StackingRegressor(
        regressors=[rf, elnet, lso, rdg, svr],
        meta_regressor=SVR(kernel='rbf'),
        #meta_regressor=Lasso(alpha=0.0005)
    )

    model.fit(train_data, y_train_values)

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
