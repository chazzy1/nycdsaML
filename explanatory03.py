import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import skew

df_train = pd.read_csv('./data/train.csv')


"""
check for skews
"""
numeric_columns = df_train.dtypes[df_train.dtypes != "object"].index


skewed_columns = df_train[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]

print(skewed_columns)




sns.distplot(df_train["SalePrice"])
plt.show()
