import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


df_train = pd.read_csv('./data/train.csv')


"""
check for NaN
"""
print(df_train.isnull().sum())
