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

from scipy.stats import skew, zscore
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

    outliers = [197, 523, 691, 854, 1182, 1298]
    print(outliers)

    z = np.abs(zscore(train_set[get_numeric_columns(train_set)]))
    row, col = np.where(z > 4)
    df = pd.DataFrame({"row": row, "col": col})
    rows_count = df.groupby(['row']).count()

    outliers = rows_count[rows_count.col > 2].index
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
        ('OutlierRemover', OutlierRemover()),
        ('NaNImputer', NaNImputer()),
        ('NaNRemover', NaNRemover()),
        #('AdditionalFeatureGenerator', AdditionalFeatureGenerator()),
        #('TypeTransformer', TypeTransformer()),
        #('ErrorImputer', ErrorImputer()),
        #('SkewFixer', SkewFixer()),
        #('Scaler', Scaler()),
        #('FeatureDropper', FeatureDropper()),
        ('Dummyfier', Dummyfier()),
    ])


    transformed_data = transform_pipeline.transform(combined_data)
    train_data = transformed_data[:train_set_rows]
    predict_data = transformed_data[train_set_rows:]

    from keras.models import Sequential
    from keras.metrics import categorical_accuracy
    from keras.layers import Dense, Activation
    from keras.optimizers import SGD

    """
    model = Sequential()
    model.add(Dense(160, input_dim=306))
    model.add(Activation('sigmoid'))
    #model.add(Dense(80))
    #model.add(Activation('softmax'))
    model.add(Dense(1))
    model.add(Activation('linear'))

    sgd = SGD(lr=0.1)
    model.compile(
        loss='mean_squared_error',
        optimizer=sgd,
        #metrics=[categorical_accuracy]
    )
    """
    model = Sequential()
    model.add(Dense(80, input_dim=301, activation='relu'))
    model.add(Dense(160, activation='relu'))
    model.add(Dense(80, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam')
    

    import time

    start = time.time()


    model.fit(train_data, y_train_values, epochs=2000, batch_size=64)

    print("\nTime elasped: {} seconds\n".format(time.time() - start))


    ensembled = np.expm1(model.predict(predict_data).reshape(-1))
    print(ensembled)


    """
    score = model.evaluate(
        mnist.validation.images,
        mnist.validation.labels
    )

    print(score)
    """

    """
    export submission data
    """

    submission = pd.DataFrame({
        "Id": test_set_id,
        "SalePrice": ensembled
    })
    submission.to_csv('submission_keras.csv', index=False)


if __name__== "__main__":
    main()
