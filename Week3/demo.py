#%%
from tracemalloc import start
import numpy as np 
import pandas as pd 
import datetime as dt 
import investpy
#%%
start = "01/01/2017"
end  = dt.datetime.now().strftime("%d/%m/%Y")
#Giá cổ phiếu bất động sản
DXG = investpy.get_stock_historical_data(stock="DXG",country='vietnam',from_date=start,to_date=end)
DXG.head()
# %%
pd.DataFrame(DXG).to_csv("DXG.csv")
# %%
