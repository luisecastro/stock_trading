# stock_trading

If the market is inefficient, then there is a posibility to profit from it. There are many ways this can be achieved either by investing or trading. To be able to have an insight of the market trends there's the need to analyse the stock, company information, etc.

The [stock_market.ipynb](https://github.com/luisecastro/stock_trading/blob/master/stock_market.ipynb) file is a walkthrough (yet to be completed) to this market environment, it starts by interfacing with the Yahoo Finance, selecting the symbols and dates to analyse, and creating pandas dataframes with information as adjusted close, daily return, etc. A careful selection of the market indicators is necesary for later applying machine learning algorithms. 

Other algorithms look to create the best portfolios depending on minimizing risk, maximizing profit or achieving the best Sharpe Ratio. They can also focus on minimizing market impact portfolio beta = 0, while maximizing alpha.

Predictions we could have time ranges from seconds to months or years, from high frequency trading to careful long term investment, both resume in how can we know the position of the stock in the future.

Two python files, st.py and dev.py, st.py contains all functions to be called as a sort of library to keep the flow of the code in dev.py clean.