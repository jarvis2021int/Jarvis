# Import library

import warnings
warnings.filterwarnings("ignore")
from datetime import date
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling 
%pylab inline


import seaborn as sns
import statsmodels.api as sm

# Import excel data file
data = pd.read_excel("Time Series Sample.xlsx") 
data.head()

# To check the shape of data like number of rows and columns
data.shape

# create the new data set of volume varible 
df = data[['XNSE/BIOCON/VOLUME','XNSE/DMART/VOLUME','XNSE/COLPAL/VOLUME','XNSE/RELIANCE/VOLUME','XNSE/INFY/VOLUME','XNSE/MANAPPURAM/VOLUME','XNSE/ICICIBANK/VOLUME','XNSE/HDFCAMC/VOLUME','XNSE/SYNGENE/VOLUME','XNSE/TITAN/VOLUME']]
df.head()

# caluulate the mean of each stock of volume
MEAN = df.mean()
MEAN

# pie-chart of volume mean

plt.figure(figsize=(12,8))
df.mean()[:10].plot(kind = "pie")
plt.title("Stock Volume Price", size = 18)
plt.show()

# create the new data frame for closing price of each stocks
df1 = data[["DATE",'XNSE/BIOCON/CLOSE',"DATE.1",'XNSE/DMART/CLOSE',"DATE.2",'XNSE/COLPAL/CLOSE',"DATE.3",'XNSE/RELIANCE/CLOSE',"DATE.4",'XNSE/INFY/CLOSE',"DATE.5",'XNSE/MANAPPURAM/CLOSE',"DATE.6",'XNSE/ICICIBANK/CLOSE',"DATE.7",'XNSE/HDFCAMC/CLOSE',"DATE.8",'XNSE/SYNGENE/CLOSE',"DATE.9",'XNSE/TITAN/CLOSE']]
df1.head()

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(441)
ax2 = fig.add_subplot(442)
ax3 = fig.add_subplot(443)
ax4 = fig.add_subplot(444)
ax5 = fig.add_subplot(445)
ax1.plot(df1['XNSE/BIOCON/CLOSE'])
ax1.set_title("BIOCON")
ax2.plot(df1['XNSE/DMART/CLOSE'])
ax2.set_title("DMART")
ax3.plot(df1['XNSE/COLPAL/CLOSE'])
ax3.set_title("COLPAL")
ax4.plot(df1['XNSE/RELIANCE/CLOSE'])
ax4.set_title("RELIANCE")
ax5.plot(df1['XNSE/INFY/CLOSE'])
ax5.set_title("INFY")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12,8))

ax6 = fig.add_subplot(441)
ax7 = fig.add_subplot(442)
ax8 = fig.add_subplot(443)
ax9 = fig.add_subplot(444)
ax10 = fig.add_subplot(445)

ax6.plot(df1['XNSE/MANAPPURAM/CLOSE'])
ax6.set_title("MANAPPURAM")
ax7.plot(df1['XNSE/ICICIBANK/CLOSE'])
ax7.set_title("ICICIBAN")
ax8.plot(df1['XNSE/HDFCAMC/CLOSE'])
ax8.set_title("HDFCAMC")
ax9.plot(df1['XNSE/SYNGENE/CLOSE'])
ax9.set_title("SYNGENE")
ax10.plot(df1['XNSE/TITAN/CLOSE'])
ax10.set_title("TITAN")
plt.tight_layout()
plt.show()

# Data sorting by ascending order

# ceate only colsing prices of each stock
data_1 = data[["DATE",'XNSE/BIOCON/CLOSE','XNSE/DMART/CLOSE','XNSE/COLPAL/CLOSE','XNSE/RELIANCE/CLOSE','XNSE/INFY/CLOSE']]
data_1 = data_1.sort_values(by =["DATE"],ascending=True, ignore_index=True)
data_1.head()

df2 = data_1[['XNSE/BIOCON/CLOSE','XNSE/DMART/CLOSE','XNSE/COLPAL/CLOSE','XNSE/RELIANCE/CLOSE','XNSE/INFY/CLOSE']]
df2.head()



data_2 = data[["DATE.5",'XNSE/MANAPPURAM/CLOSE','XNSE/ICICIBANK/CLOSE','XNSE/HDFCAMC/CLOSE','XNSE/SYNGENE/CLOSE','XNSE/TITAN/CLOSE']]
data_2 = data_2.sort_values(by =["DATE.5"],ascending=True, ignore_index=True)
data_2.head()

data_1 = data_1.drop("DATE", axis = 1)

data_2 = data_2.drop("DATE.5", axis = 1)

df2 = pd.concat([data_1, data_2], axis = 1)
df2.head()

# Mean of closing price 

df2.mean()

plt.figure(figsize=(12,8))
df2.mean()[:10].plot(kind = "pie")
plt.title("Stock closing price", size = 18)
plt.show()


Variance
$$s^2 = \frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N-1}$$
Standard Deviation (Volatility)
$$s = \sqrt{\frac{\sum_{i=1}^N (x_i - \bar{x})^2}{N-1}}$$

# Calculating the daily  returns for individual stock

#daily_return.to_csv("daily_return.csv")

daily_return = df2.pct_change()
daily_return.head()

daily_return.sum()

# mean of daily return
daily_return.mean()

# Standard deviation in % formate

daily_return.std()*100



plt.figure(figsize=(12,8))
(daily_return.std()*100)[:10].plot(kind = "pie")
plt.title("Stock return price in %", size = 18)
plt.show()

plt.figure()
daily_return[['XNSE/BIOCON/CLOSE','XNSE/DMART/CLOSE','XNSE/COLPAL/CLOSE','XNSE/RELIANCE/CLOSE','XNSE/INFY/CLOSE']].plot(figsize = (13, 8))
plt.ylabel("Daily return of all stock",size = 20)
plt.xlabel("Day", size = 20)
plt.title("Daily stock return", size =25)
plt.show()

plt.figure()
daily_return[['XNSE/MANAPPURAM/CLOSE','XNSE/ICICIBANK/CLOSE','XNSE/HDFCAMC/CLOSE','XNSE/SYNGENE/CLOSE','XNSE/TITAN/CLOSE']].plot(figsize = (13, 8))
plt.ylabel("Daily return of all stock",size = 20)
plt.xlabel("Day", size = 20)
plt.title("Daily stock return",size = 25)
plt.show()



var_90 = daily_return.quantile(0.1)
var_95 = daily_return.quantile(1-0.95)
var_99 = daily_return.quantile(1-0.99)

var_90


var_95

var_99

var = pd.DataFrame({"VAR_90":var_90, "VAR_95":var_95, "VAR_99": var_99})
var



stocks = df2[['XNSE/BIOCON/CLOSE', 'XNSE/DMART/CLOSE', 'XNSE/COLPAL/CLOSE',
       'XNSE/RELIANCE/CLOSE', 'XNSE/INFY/CLOSE','XNSE/MANAPPURAM/CLOSE',
       'XNSE/ICICIBANK/CLOSE','XNSE/SYNGENE/CLOSE','XNSE/TITAN/CLOSE']]
return_stock = stocks.pct_change()


return_stock.dropna(inplace=True)

return_stock.shape

return_stock.head()

#from pandas_datareader import data as pdr

nifty_return = pd.read_csv("^NSEI.csv")
#nifty_return =market_return
# [963:984]
nifty_return

nifty_return = nifty_return["Close"].pct_change()

nifty_return.dropna(inplace=True)
nifty_return = nifty_return[85:]
nifty_return.shape

nifty_return.dropna(inplace=True)
nifty_return.shape

x1 = return_stock
y = nifty_return

x1= return_stock.values
y= nifty_return.values

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()
results.summary()

from statsmodels import regression

x1 = return_stock["XNSE/BIOCON/CLOSE"] 
y = nifty_return

x1= x1.values
y= nifty_return.values

x1 = sm.add_constant(x1)
results_1 = regression.linear_model.OLS(y,x1).fit()
print("alpha",results_1.params[0])
print("beta",results_1.params[1])


x2 = return_stock["XNSE/DMART/CLOSE"] 
y = nifty_return

x2= x2.values
y= nifty_return.values

x2 = sm.add_constant(x2)
results_2 = regression.linear_model.OLS(y,x2).fit()
print("alpha",results_2.params[0])
print("beta",results_2.params[1])


x3 = return_stock["XNSE/COLPAL/CLOSE"] 
y = nifty_return

x3= x3.values
y= nifty_return.values

x3 = sm.add_constant(x3)
results_3 = regression.linear_model.OLS(y,x3).fit()
print("alpha",results_3.params[0])
print("beta",results_3.params[1])


x4 = return_stock["XNSE/RELIANCE/CLOSE"] 
y = nifty_return

x4= x4.values
y= nifty_return.values

x4 = sm.add_constant(x4)
results_4 = regression.linear_model.OLS(y,x4).fit()
print("alpha",results_4.params[0])
print("beta",results_4.params[1])


x5 = return_stock["XNSE/INFY/CLOSE"] 
y = nifty_return

x5= x5.values
y= nifty_return.values

x5 = sm.add_constant(x5)
results_5 = regression.linear_model.OLS(y,x5).fit()
print("alpha",results_5.params[0])
print("beta",results_5.params[1])


x6 = return_stock["XNSE/MANAPPURAM/CLOSE"] 
y = nifty_return

x6= x6.values
y= nifty_return.values

x6 = sm.add_constant(x6)
results_6 = regression.linear_model.OLS(y,x6).fit()
print("alpha",results_6.params[0])
print("beta",results_6.params[1])


x7 = return_stock["XNSE/ICICIBANK/CLOSE"] 
y = nifty_return

x7= x7.values
y= nifty_return.values

x7 = sm.add_constant(x7)
results_7 = regression.linear_model.OLS(y,x7).fit()
print("alpha",results_7.params[0])
print("beta",results_7.params[1])


x8 = return_stock["XNSE/SYNGENE/CLOSE"] 
y = nifty_return

x8= x8.values
y= nifty_return.values

x8 = sm.add_constant(x8)
results_8 = regression.linear_model.OLS(y,x8).fit()
print(results_8.params[0])
print(results_8.params[1])



x9 = return_stock["XNSE/BIOCON/CLOSE"] 
y = nifty_return

x9= x9.values
y= nifty_return.values

x9 = sm.add_constant(x9)
results_9 = regression.linear_model.OLS(y,x9).fit()
print("alpha",results_9.params[0])
print("beta",results_9.params[1])


results.params[9]

Alpha_beta = pd.DataFrame({"Stock name":["BIOCON","DMART","COLPAL","RELIANCE","INFY","MANAPPURAM","ICICIBANK","SYNCENE","TITAN"],
                       "Regression_Alpha":[results_1.params[0],results_2.params[0],results_3.params[0],results_4.params[0],results_5.params[0],results_6.params[0],results_7.params[0],results_8.params[0],results_9.params[0]],
                  "Regression_beta":[results_1.params[1],results_2.params[1],results_3.params[1],results_4.params[1],results_5.params[1],results_6.params[1],results_7.params[1],results_8.params[1],results_9.params[1]]})
                   
Alpha_beta

# Calulating the avarge ture Range

# open and and close price two data frame

df3 = data[['XNSE/BIOCON/OPEN','XNSE/DMART/OPEN','XNSE/COLPAL/OPEN','XNSE/RELIANCE/OPEN','XNSE/INFY/OPEN','XNSE/MANAPPURAM/OPEN','XNSE/ICICIBANK/OPEN','XNSE/HDFCAMC/OPEN','XNSE/SYNGENE/OPEN','XNSE/TITAN/OPEN']]
df4 = data[[ 'XNSE/DMART/CLOSE','XNSE/DMART/CLOSE','XNSE/COLPAL/CLOSE','XNSE/RELIANCE/CLOSE','XNSE/INFY/CLOSE','XNSE/MANAPPURAM/CLOSE','XNSE/ICICIBANK/CLOSE','XNSE/HDFCAMC/CLOSE','XNSE/SYNGENE/CLOSE','XNSE/TITAN/CLOSE']]
df3.head()


df4.head()

df4.fillna(0,inplace=True)
df3.fillna(0,inplace=True)

# formula for the calculate ATR
data=(df4.values+df3.values)/2

# This is data frame for open and close divided by two 

df5=pd.DataFrame(data, columns =("BIOCON","DMART","COLPAL","RELIANCE","INFY","MANAPPURAM","ICICIBANK","HDFCAMC","SYNGENE","TITAN") )
df5.head()

df5.shape

# This is the mean of ATR of 1000 observation

mean_ATR = df5.mean()
mean_ATR

ten_day_ATR = df5[991:].mean()
ten_day_ATR

fifteen_day = df5[995:].mean()
fifteen_day 

ATR = pd.DataFrame({"Mean_ATR":mean_ATR, "LAST_TEN_ATR": ten_day_ATR, "Last_fifteen_ATR":fifteen_day})
ATR

# Import excel data file
data = pd.read_excel("Time Series Sample.xlsx")
data.head()



def atr(data, n=14):
    data = data.copy()
    high = data['XNSE/BIOCON/HIGH']
    low = data['XNSE/BIOCON/LOW']
    close = data['XNSE/BIOCON/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_1 = atr(data)
print(ATR_1.mean())

ten_1 = ATR_1[991:].mean()
print(ten_1)

fifteen_1 = ATR_1[996:].mean()
print(fifteen_1)

def atr(data):
    data = data.copy()
    high = data['XNSE/DMART/HIGH']
    low = data['XNSE/DMART/LOW']
    close = data['XNSE/DMART/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_2 = atr(data)
print(ATR_2.mean())

ten_2 = ATR_2[991:].mean()
print(ten_2)

fifteen_2 = ATR_2[996:].mean()
print(fifteen_2)

def atr(data):
    data = data.copy()
    high = data['XNSE/COLPAL/HIGH']
    low = data['XNSE/COLPAL/LOW']
    close = data['XNSE/COLPAL/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_3 = atr(data)
print(ATR_3.mean())

ten_3 = ATR_3[991:].mean()
print(ten_3)

fifteen_3 = ATR_3[996:].mean()
print(fifteen_3)

def atr(data):
    data = data.copy()
    high = data['XNSE/RELIANCE/HIGH']
    low = data['XNSE/RELIANCE/LOW']
    close = data['XNSE/RELIANCE/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_4 = atr(data)
print(ATR_4.mean())

ten_4 = ATR_4[991:].mean()
print(ten_4)

fifteen_4 = ATR_4[996:].mean()
print(fifteen_4)

def atr(data):
    data = data.copy()
    high = data['XNSE/INFY/HIGH']
    low = data['XNSE/INFY/LOW']
    close = data['XNSE/INFY/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr =tr
    return atr

ATR_5 = atr(data)
print(ATR_5.mean())

ten_5 = ATR_5[991:].mean()
print(ten_5)

fifteen_5 = ATR_5[996:].mean()
print(fifteen_5)

def atr(data, n=14):
    data = data.copy()
    high = data['XNSE/MANAPPURAM/HIGH']
    low = data['XNSE/MANAPPURAM/LOW']
    close = data['XNSE/MANAPPURAM/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_6 = atr(data)
print(ATR_6.mean())

ten_6 = ATR_6[991:].mean()
print(ten_6)

fifteen_6 = ATR_6[996:].mean()
print(fifteen_6)

def atr(data, n=14):
    data = data.copy()
    high = data['XNSE/ICICIBANK/HIGH']
    low = data['XNSE/ICICIBANK/LOW']
    close = data['XNSE/ICICIBANK/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_7 = atr(data)
print(ATR_7.mean())

ten_7 = ATR_7[991:].mean()
print(ten_7)

fifteen_7 = ATR_7[996:].mean()
print(fifteen_7)

def atr(data):
    data = data.copy()
    high = data['XNSE/HDFCAMC/HIGH']
    low = data['XNSE/HDFCAMC/LOW']
    close = data['XNSE/HDFCAMC/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_8 = atr(data)
print(ATR_8.mean())

ten_8 = ATR_8[991:].mean()
print(ten_8)

fifteen_8 = ATR_8[996:].mean()
print(fifteen_8)

def atr(data):
    data = data.copy()
    high = data['XNSE/SYNGENE/HIGH']
    low = data['XNSE/SYNGENE/LOW']
    close = data['XNSE/SYNGENE/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_9 = atr(data)
print(ATR_9.mean())

ten_9 = ATR_9[991:].mean()
print(ten_9)

fifteen_9 = ATR_9[991:].mean()
print(fifteen_9)

def atr(data):
    data = data.copy()
    high = data['XNSE/TITAN/HIGH']
    low = data['XNSE/TITAN/LOW']
    close = data['XNSE/TITAN/CLOSE']
    data['tr0'] = abs(high - low)
    data['tr1'] = abs(high - close.shift())
    data['tr2'] = abs(low - close.shift())
    tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
    atr = tr
    return atr

ATR_10 = atr(data)
print(ATR_10.mean())

ten_10 = ATR_10[991:].mean()
print(ten_10)

fifteen_10 = ATR_10[996:].mean()
print(fifteen_10)

ATR = pd.DataFrame({"Stock name":["BIOCON","DMART","COLPAL","RELIANCE","INFY","MANAPPURAM","ICICIBANK","HDFCAMC","SYNCENE","TITAN"],
                  "Mean_ATR_1000":[ATR_1.mean(),ATR_2.mean(),ATR_3.mean(),ATR_4.mean(),ATR_5.mean(),ATR_6.mean(),ATR_7.mean(),ATR_8.mean(),ATR_9.mean(),ATR_10.mean()],
                  "ten_day_ATR":[ten_1,ten_2,ten_3,ten_4,ten_5,ten_6,ten_7,ten_8,ten_9,ten_10],
                    "fifteen_day_ATR":[fifteen_1,fifteen_2,fifteen_3,fifteen_4,fifteen_5,fifteen_6,fifteen_7,fifteen_8,fifteen_9,fifteen_10]})

ATR