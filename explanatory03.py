import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import skew
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('seaborn')
from scipy.stats import norm, skew
from scipy import stats
import numpy as np
import seaborn as sns
from pylab import *
sns.set()
pd.set_option('max_columns', None)
#Data loading
train_set = pd.read_csv('./data/train.csv')
test_set = pd.read_csv('./data/test.csv')
df_train = pd.read_csv('./data/train.csv')
combined_data = pd.concat((train_set, test_set), sort=False)

X = combined_data



"""
check for skews
"""
numeric_columns = df_train.dtypes[df_train.dtypes != "object"].index


skewed_columns = df_train[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewed_columns = skewed_columns[abs(skewed_columns) > 0.75]

print(skewed_columns)




sns.distplot(df_train["SalePrice"])
#plt.show()



subplots_adjust(hspace=0.000)
number_of_subplots=3
columns = X.dtypes[X.dtypes != "object"].index
plots_count = int(columns.shape[0])


fig = plt.figure( figsize=(plots_count*1,200) )
v=1
for i in columns:
    ax1 = subplot(columns.shape[0],1,v)
    stats.probplot(X[i], dist="norm", plot= fig.add_subplot(ax1))
    v = v+1

plt.show()