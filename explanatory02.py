import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df_train = pd.read_csv('./data/train.csv')


"""
check for NaN
"""

nansum = df_train.isnull().sum()
nansum = pd.DataFrame({'col': nansum.index, 'sum': nansum.values})
nansum = nansum[nansum['sum'] > 0]
print(nansum)


