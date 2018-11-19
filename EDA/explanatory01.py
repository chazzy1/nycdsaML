import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df_train = pd.read_csv('./data/train.csv')



"""
check for correlation
"""
corr_matrix = df_train.corr()

plt.subplots(figsize=(12, 12))
sns.heatmap(corr_matrix, vmax=.8, square=True)


largest_corr_cols = corr_matrix.nlargest(10, 'SalePrice')
print(largest_corr_cols)

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], height=1)



plt.show()