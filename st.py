# -*- coding: utf-8 -*-
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression


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
    
    return df



# Function to fill NA's and NaN's in the data frame, it searches for them and first
# tries to fill them forward, using the last available value, then it backward fills
# whatever is missing with the first available value

def fillna(df):
    ndf = df.copy()
    ndf.fillna(method='ffill',inplace='TRUE')
    ndf.fillna(method='bfill',inplace='TRUE')
    return ndf



# Divides the complete dataframe by the value of the first row, making this row
# start in 1.0 for all columns and continue as a rate of this first value

def normalize(df):
    ndf = df.copy()
    return ndf/ndf.ix[0,:]



# Computes rolling statistics, mean and std along with Bollinger Bands(r)
# it receives a dataframe, a symbol and the size of the window to be used

def rolling(df,window=20):
    columns = list()
    stats = ['_raw','_sma','_ema','_std','_lbb','_hbb']

    for i in df.columns:
        for j in stats:
            columns.append(i+j)

    ndf = pd.DataFrame(index=df.index,columns=columns)

    for i in df.columns:
            ndf[i+stats[0]] = df[i]
            ndf[i+stats[1]] = df[i].rolling(window=window,center=False).mean()
            ndf[i+stats[2]] = df[i].ewm(span=window).mean()
            ndf[i+stats[3]] = df[i].rolling(window=window,center=False).std()
            ndf[i+stats[4]] = ndf[i+'_sma']+2*ndf[i+'_std']
            ndf[i+stats[5]] = ndf[i+'_sma']-2*ndf[i+'_std']
    
    return ndf.ix[window:]




# Takes a data frame and returns only the columns between the dates specified
# of the closest dates to them

def time_slice(df,start,end):
    ndf = df.copy()
    start = ndf.index.searchsorted(start)
    end = ndf.index.searchsorted(end)
    return ndf.ix[start:end]



# Calculates the daily return, it takes the data frame and divides it by the 
# previous n days, and substracts one, resulting in the rate of variation from
# the n day to the next one

def dailyReturn(df,n=1):
    ndf = df.copy()
    ndf[n:] = (df[n:]/df[:-n].values)-1
    ndf.ix[range(n),:] = 0
    return ndf



# Similar to daily return, it calculates the rate of return but to the first
# date available on the dataframe

def cumReturn(df):
    ndf = df.copy()
    return ndr/ndr.ix[0]-1



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

    n = portfolio.shape[1]

    for i in range(n):
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
# alpha (always have an advantage of the market)

def riskPro(allocation,betas,alphas):
    return np.sum(np.multiply(betas,allocation)),np.sum(np.multiply(alphas,allocation))



# Filter function to extract only certain symbols or certain stats

def filter(df,symbol):
    length = len(symbol)
    columns = list()
    for i in range(len(df.columns)):
        if df.columns[i][0:length]==symbol or df.columns[i][-length:]==symbol:
            columns.append(i)

    return df[df.columns[columns]]



# Machine Learning Classification Algorithms

# Random Forest
def rfClass(X_train,y_train,n=1000,c='gini'):
    clf = RandomForestClassifier(n_estimators=n,criterion=c,n_jobs=-1)
    clf.fit(X_train,y_train)
    return clf

# K Nearest Neighbors
def knnClass(X_train,y_train,n=5,w='uniform'):
    clf = KNeighborsClassifier(n_neighbors=n,weights=w,n_jobs=-1)
    clf.fit(X_train,y_train)
    return clf

# Support Vector Machine
def svmClass(X_train,y_train,c=1,g='auto',k='rbf'):
    clf = SVC(C=c,kernel=k,gamma=g,n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

# Adaptative Boosting
def aboostClass(X_train,y_train,n=1000):
    clf = AdaBoostClassifier(n_estimators=n)
    clf.fit(X_train, y_train)   
    return clf

# Gradient Tree Boosting
def gtbClass(X_train,y_train,n=1000):
    clf = GradientBoostingClassifier(n_estimators=n)
    clf.fit(X_train, y_train)
    return clf

# Quadratic Discriminant Analysis
def qdaClass(X_train,y_train,tiny=0.0001):
    X_train[X_train<tiny] = tiny
    y_train[y_train<tiny] = tiny
    
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    return clf

# Logistic Regression
def lrClass(X_train,y_train):
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf