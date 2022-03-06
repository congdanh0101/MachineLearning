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
# %%
raw_data.plot(kind="scatter",y = "Close",x = "Volume",alpha = .35,figsize=(20,10))
plt.axis([100000,10000000,0,50000])
plt.legend()
# %%
from pandas.plotting import scatter_matrix
features_to_plot = ['Close']
scatter_matrix(raw_data[features_to_plot],figsize=(20,10))
plt.show()
# %%
raw_data.hist(figsize=(20,10))
plt.rcParams['xtick.labelsize']=10
plt.rcParams['ytick.labelsize']=10
plt.tight_layout()
plt.savefig("figures/hist_raw_data.png",format='png',dpi=500)
plt.show()

# %%
corr_matrix = raw_data.corr()

# %%
import cufflinks as cf
cf.go_offline()
# %%
raw_data.drop("Currency",axis=1,inplace=True)
raw_data[['Open','High','Low','Close']].loc['2021-01-01':'2022-01-01'].iplot(kind='candle')

# %%
DXG = investpy.get_stock_historical_data(stock="DXG",country='vietnam',from_date="01/01/2021",to_date="01/01/2022")
DXG[['Open','High','Low','Close']].loc['2021-05-01':'2022-01-01'].iplot(kind='candle')

# %%
search_result = investpy.search_quotes(text='dat xanh', products=['stocks'],countries=['vietnam'],n_results=1)
# %%
recent_data = search_result.retrieve_recent_data()
print(recent_data.head())
# %%
import datetime as dt
enddate = dt.datetime.now().strftime("%d/%m/%Y")
historical_data = search_result.retrieve_historical_data(from_date='01/01/2020',to_date=enddate)
historical_data.tail()
# %%
information_search = search_result.retrieve_information()
print(information_search['weekRange'])

# %%
default_currency = search_result.retrieve_currency()
print(default_currency)
# %%
technical_indicators = search_result.retrieve_technical_indicators(interval='weekly')
print(technical_indicators)
# %%
data = investpy.moving_averages(name='CTG',country='vietnam',product_type='stock',interval='weekly')
data.head()
# %%
