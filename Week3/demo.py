#%%
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import datetime as dt 
import investpy
import seaborn as sns
#%%
start = "01/01/2017"
end  = dt.datetime.now().strftime("%d/%m/%Y")
#Giá cổ phiếu bất động sản
DXG = investpy.get_stock_historical_data(stock="DXG",country='vietnam',from_date=start,to_date=end)
NLG = investpy.get_stock_historical_data(stock="NLG",country='vietnam',from_date=start,to_date=end)
KDH = investpy.get_stock_historical_data(stock="KDH",country='vietnam',from_date=start,to_date=end)
#%%
DXG.drop("Currency",axis=1,inplace=True)
NLG.drop("Currency",axis=1,inplace=True)
KDH.drop("Currency",axis=1,inplace=True)
# %%
pd.DataFrame(DXG).to_csv("DXG.csv")
pd.DataFrame(NLG).to_csv("NLG.csv")
pd.DataFrame(KDH).to_csv("KDH.csv")
# %%
list_RealEstate = ['DXG','NLG','KDH']
RealEstate_Stock = pd.concat([DXG,NLG,KDH],axis=1,keys=list_RealEstate)
RealEstate_Stock.columns.names = ['Stock','Info']
RealEstate_Stock.head()
# %%
RealEstate_Stock.xs(key='Close',axis=1,level='Info').max()
# %%
change_percent_close = pd.DataFrame()
for name in list_RealEstate:
    change_percent_close[name] = RealEstate_Stock[name]['Close'].pct_change()*100
change_percent_close.dropna(inplace=True)
change_percent_close.head()
# %%
sns.pairplot(change_percent_close)
# %%
from pandas.plotting import scatter_matrix
features_to_plot = ['Open','High','Low','Close','Volume']
scatter_matrix(DXG[features_to_plot],figsize=(10,8))
plt.show()
# %%
plt.figure(figsize=(12,8))
sns.distplot(change_percent_close.loc['2021-01-01':'2022-01-01']['DXG'],color='green',bins=100)
# %%
for stock_name in list_RealEstate:
    RealEstate_Stock[stock_name]['Close'].plot(figsize=(20,10),label=stock_name)
plt.legend()
# %%
import cufflinks as cf
cf.go_offline()
RealEstate_Stock.xs(key='Close',axis = 1,level='Info').iplot(title='Giá đóng cửa',xTitle = 'Năm',yTitle='Giá')
# %%
RealEstate_Stock['DXG']['Close'].plot(figsize=(20,10))
plt.show()
# %%
DXG['Close'].loc['01/01/2021':'01/01/2022'].ta_plot(study='boll',periods=14)
# %%
DXG[['Open','High','Low','Close']].loc['06/01/2021':end].iplot(kind='candle')
# %%
cf.datagen.ohlc()
DXG_plot = cf.QuantFig(DXG.loc['10/01/2021':end],legend='Top',name='Nến',color='red')
DXG_plot.add_sma([20,50],color=['#12345a','blue'])
DXG_plot.iplot()


# %%
a = RealEstate_Stock.xs(key='Close',axis=1,level='Info').mean()

# %%
meanStock = 0
count = 0
for i in range(0,len(a)):
    count+=1
    meanStock+=a[i]
print((meanStock/count).__round__(2))
# %%
data = investpy.search_quotes(text='DXG',products=['stocks'],countries=['vietnam'],n_results=1)
print(data)
# %%
technical = data.retrieve_technical_indicators(interval='daily')
print(technical)

# %%
