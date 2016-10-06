# -*- coding: utf-8 -*-
import st
import matplotlib.pyplot as plt

# Start with 4 of biggest tech (and overall) companies:
# Apple, Google (Alphabet), Microsoft, Amazon
# 
# And use the indicators of the SPY500 
# symbols = ['GOOGL','AAPL','MSFT','AMZN','SPY']# ['^IXIC','^GDAXI','^FCHI','^N225','^HSI','^AXJO','^DJI']
symbols = ['GOOGL','SPY']


# Download 5 years worth of historic data
dates = ['2011-09-30','2016-09-30']


# Call function to download data
st.stock_dl(symbols,dates[0],dates[1])


# Start selecticting the dates and adj_close columns, the get_data can always be called if needed
# for another column
# columns = ['volume','adj_close','high','low','date','close','open']
columns = ['adj_close','date']
dat = st.get_data(symbols,columns,dates[0],dates[1])
dat.columns = symbols


# Fill the missing data with back and fwd fill
dat = st.fillna(dat)


# Normalize data, start with one's in all columns to check variation over the time
# and have all columns with similar values
norm = st.normalize(dat)


# Get rolling statistics, including rolling mean, rolling exponential mean, rolling std and bollinger bands(R)
roll = st.rolling(norm)
# print st.filter(st.rolling(norm),'sma')
print roll

# Get daily returns, that is the rate of increase or decrease of the stock, number of
# days of difference can be specified
dr_1 = st.dailyReturn(norm)
dr_2 = st.dailyReturn(norm,2)
dr_3 = st.dailyReturn(norm,3)

