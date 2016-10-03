import yahoo_finance
import csv
import os.path
import os
import pandas as pd
import numpy as np
import scipy.optimize as spo



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



def symbol_to_path(symbol,base_dir="stock"):
    return os.path.join(base_dir,"{}.csv".format(str(symbol)))



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



def fillna(df):
    df.fillna(method='ffill',inplace='TRUE')
    df.fillna(method='bfill',inplace='TRUE')
    return df



def normalize(df):
    return df/df.ix[0,:]



def rolling(df,symbol,window=20):
    temp = pd.DataFrame(index=df.index,columns=['mean','std','lbb','hbb'])
    temp['mean'] = df.rolling(window=window,center=False).mean()
    temp['std'] = df.rolling(window=window,center=False).std()
    temp['lbb'] = temp['mean']+2*temp['std']
    temp['hbb'] = temp['mean']-2*temp['std']
    return temp


def time_slice(df,start,end):
    start = df.index.searchsorted(start)
    end = df.index.searchsorted(end)
    return df.ix[start:end]



def dailyReturn(df):
    df[1:] = (df[1:]/df[:-1].values)-1
    df.ix[0,:] = 0
    return df



def cumReturn(df):
    return dr/dr.ix[0]-1



def fitPol(df,symbol1,symbol2='SPY',n=1):
    return np.polyfit(df[symbol1],df[symbol2],n)



def stats(df,drf=0,samples=252):
    temp = pd.DataFrame(index=df.columns.values,columns=['mean','std','kurtosis','sharpe'])
    temp['mean'] = df.mean()
    temp['std'] = df.std()
    temp['kurtosis'] = df.kurtosis()
    temp['sharpe'] = (samples**0.5)*((df.mean()-drf)/df.std())
    return temp



def portVal(df,allocation):
    return np.sum(np.multiply(df/df.ix[0,:],allocation.values),axis=1)



def sharpe(allocation,portfolio,drf=0,samples=252):
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*np.sqrt(samples)*(portfolio.mean()-drf)/portfolio.std()



def risk(allocation,portfolio):
    portfolio = (portfolio*allocation).sum(axis=1)
    return portfolio.std()



def profit(allocation,portfolio):
    portfolio = (portfolio*allocation).sum(axis=1)
    return -1.*portfolio.sum()



def portOpt(f,portfolio):
    cons = ({'type':'eq','fun':lambda x: 1-sum(x)}) # sum(abs(x))
    allocation = list()
    bounds = list()

    for i in range(portfolio.shape[1]):
        allocation.append(1./n)
        bounds.append((0,1)) # (-1,1)


    return spo.minimize(f,allocation,args=(portfolio,),method='SLSQP',
        options={'disp':True},bounds=bounds,constraints=cons)


def preVal(fv,ir,i):
    return fv/(1+ir)**i



def portLin(df):
    symbols = df.columns.values
    temp = (pd.DataFrame(index=symbols,columns=['beta','alpha']))
    
    for i in range(symbols.shape[0]):
        temp.ix[i,0], temp.ix[i,1] = fitPol(df,symbols[i])
    
    return temp
















