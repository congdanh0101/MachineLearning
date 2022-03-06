#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from statistics import mean
from sklearn.model_selection import KFold   
import joblib 
import investpy
#%%
raw_data = pd.read_csv("DXG.csv")
#%%
print("_________________DATASET DXG INFO________________")
print(raw_data.info())
#%%
raw_data.head()#Lấy 5 dòng đầu tiên
# %%
print("________________SOME DATA EXAMPLES________________")
raw_data.head(10)#Lấy 10 dòng đầu tiên
# %%
raw_data.drop("Currency",axis=1,inplace=True)#Drop column Currency
#%%
raw_data.describe()
# %%
raw_data.tail()# Lấy 5 dòng cuối cùng
# %%
raw_data.tail(10) #Lấy 10 dòng cuối cùng
#%%
raw_data.loc[[0,5,10],['Open','Close']]
# %%
raw_data.plot(kind="scatter",y='Close',x='Volume',alpha=.35)
plt.show()
# %%
change_percent_close = pd.DataFrame()
change_percent_close = raw_data['Close'].pct_change()
change_percent_close.dropna()
change_percent_close.head()
# %%
from pandas.plotting import scatter_matrix
features_to_plot=['Open','Close','Volume']
scatter_matrix(raw_data[features_to_plot],figsize=(12,8))
plt.show()