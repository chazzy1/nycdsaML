
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')

from sklearn.model_selection import cross_val_score, KFold

n_folds = 5

def rmsle_cv(model,X,Y):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X)
    rmse = np.sqrt(-cross_val_score(model,
                                   X,
                                   Y,
                                   scoring="neg_mean_squared_error", cv = kf))
    return rmse