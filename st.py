# Luis Castro 2016
# This file will serve as a library of functions to be called to perform a step 
# by step analysis of the stock market

# Import libraries to be used by the functions
import yahoo_finance
import csv
import os.path
import os
import pandas as pd
import numpy as np
import scipy.optimize as spo



# Function accesses Yahoo Finance API and downloads the historical data according
# to the symbol of the stock and the date range indicated. Before doing so, checks
# if there is already a file from such symbol

def stock_dl(symbol,start,end):
    columns = ['volume','symbol','adj_close','high','low','date','close','open']
    for j in symbol:
        filepath = "stock/{}.csv".format(j)
        if not os.path.isfile(filepath): 
            
            stock = yahoo_finance.Share(j)
            history = stock.get_historical(start,end)
        
            with open(filepath, "w") as toWrite:
                writer = csv.writer(toWrite,delimiter=",")
                writer.writerow(columns)
                for i in history:
                    temp = []
                    for a in i.keys():
                        temp.append(i[a])
                    writer.writerow(temp)



# Auxiliary function, creates a path using the symbol provided

def symbol_to_path(symbol,base_dir="stock"):
    return os.path.join(base_dir,"{}.csv".format(str(symbol)))



# Create a Pandas dataframe, reading the files from the symbols specified, 
# and selecting the columns indicated along with a time range
# it uses the SPY symbol to identify the trading days

def get_data(symbols,columns,start,end):
    dates = pd.date_range(start,end)
    df = pd.DataFrame(index=dates)
    
    for symbol in symbols:
        df_temp = pd.read_csv(symbol_to_path(symbol),index_col='date',parse_dates=True,
                              usecols=columns,na_values=['nan'])
        for i in columns:
            df_temp = df_temp.rename(columns={i:symbol+'_'+i})
        df = df.join(df_temp,how='left')
    
    df = df.dropna(subset=['SPY_adj_close'])
    df.columns = symbols
    
    return df



# Function to fill NA's and NaN's in the data frame, it searches for them and first
# tries to fill them forward, using the last available value, then it backward fills
# whatever is missing with the first available value

def fillna(df):
    df.fillna(method='ffill',inplace='TRUE')
    df.fillna(method='bfill',inplace='TRUE')
    return df



# Divides the complete dataframe by the value of the first row, making this row
# start in 1.0 for all columns and continue as a rate of this first value

def normalize(df):
    return df/df.ix[0,:]



# Computes rolling statistics, mean and std along with Bollinger Bands(r)
# it receives a dataframe, a symbol and the size of the window to be used

def rolling(df,symbol,window=20):
    temp = pd.DataFrame(index=df.index,columns=['mean','std','lbb','hbb'])
    temp['mean'] = df.rolling(window=window,center=False).mean()
    temp['std'] = df.rolling(window=window,center=False).std()
    temp['lbb'] = temp['mean']+2*temp['std']
    temp['hbb'] = temp['mean']-2*temp['std']
    return temp


# Takes a data frame and returns only the columns between the dates specified
# of the closest dates to them

def time_slice(df,start,end):
    start = df.index.searchsorted(start)
    end = df.index.searchsorted(end)
    return df.ix[start:end]



# Calculates the daily return, it takes the data frame and divides it by the 
# previous date, and substracts one, resulting in the rate of variation from
# one day to the next one

def dailyReturn(df):
    df[1:] = (df[1:]/df[:-1].values)-1
    df.ix[0,:] = 0
    return df



# Similar to daily return, it calculates the rate of return but to the first
# date available on the dataframe

def cumReturn(df):
    return dr/dr.ix[0]-1



# Fits a polynomial to the data, can specify the degree of the polynomial
# usually 1, and the symbols to use as data (usually one of them is the market SPY)

def fitPol(df,symbol1,symbol2='SPY',n=1):
    return np.polyfit(df[symbol1],df[symbol2],n)



# Computes statistics for each symbol in the dataframe, mean, std, kurtosis and sharpe ratio
# risk free can be specified with drf

def stats(df,drf=0,samples=252):
    temp = pd.DataFrame(index=df.columns.values,columns=['mean','std','kurtosis','sharpe'])
    temp['mean'] = df.mean()
    temp['std'] = df.std()
    temp['kurtosis'] = df.kurtosis()
    temp['sharpe'] = (samples**0.5)*((df.mean()-drf)/df.std())
    return temp



# Calculates the value of a portfolio, takes the dataframe and the allocation of weights 
# and outputs a number (value of portfolio on day 1 is 1)

def portVal(df,allocation):
    return np.sum(np.multiply(df/df.ix[0,:],allocation.values),axis=1)



# The following 3 functions are used with the 4th one, as each can be used to optimize
# the portfolio according to a metric, highest sharpe ratio, lowest risk (lowest std) 
# or maximum portfolio value

def sharpe(allocation,portfolio,drf=0,samples=252):
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*np.sqrt(samples)*(portfolio.mean()-drf)/portfolio.std()



def risk(allocation,portfolio):
    portfolio = (portfolio*allocation).sum(axis=1)
    return portfolio.std()



def profit(allocation,portfolio):
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*portfolio.sum()



# This is a portfolio optimizing function, takes a function to maximize (or minimize)
# and iterates until finding the optimum value. The constrain is that all values in the
# portfolio must sum up to 1 and the values of each weight must range from 0 to 1

def portOpt(f,portfolio):
    cons = ({'type':'eq','fun':lambda x: 1-sum(x)}) # sum(abs(x))
    allocation = list()
    bounds = list()

    for i in range(portfolio.shape[1]):
        allocation.append(1./n) # used random?
        bounds.append((0,1)) # (-1,1)


    return spo.minimize(f,allocation,args=(portfolio,),method='SLSQP',
        options={'disp':True},bounds=bounds,constraints=cons)


# Calculate present value with future value, interest rate and time

def preVal(fv,ir,i):
    return fv/(1+ir)**i



# Function to calculate the alphas and betas of all the symbols in the
# dataframe, they are calculated against SPY

def portLin(df):
    symbols = df.columns.values
    temp = (pd.DataFrame(index=symbols,columns=['beta','alpha']))
    
    for i in range(symbols.shape[0]):
        temp.ix[i,0], temp.ix[i,1] = fitPol(df,symbols[i])
    
    return temp


# Calculates de alphas and betas of an entire portfolio, the aim here would be
# to minimize beta (make the portfolio independent of the market) while maximizing
# alpha (always have and advantage of the market)

def riskPro(allocation,betas,alphas):
    return np.sum(np.multiply(betas,allocation)),np.sum(np.multiply(alphas,allocation))

















