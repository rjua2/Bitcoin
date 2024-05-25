# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:50:54 2024

@author: idpad
"""
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime

import plotly.express as px

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import datetime
import riskfolio as rp

#%%
#First, let´s retrieve data from the top 100 
#       list of all active cryptocurrencies with latest market data.
coinbase= "53110df6-1387-4165-b789-523614f455dc" # API´s key

headers = {
    "X-CMC_PRO_API_KEY": coinbase,
    "Accepts": "application/json"
}

# Data from the top 100 cryptos
params = {
    "start": "1",
    "limit": "29",
    "convert": "USD"
}

url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"

# Fetch JSON data
json_data = requests.get(url, params=params, headers=headers).json()

# Extract the relevant data
coins = json_data["data"]

# Create a DataFrame
df = pd.DataFrame.from_dict(coins, orient="columns")

# Display the DataFrame
df.head()

#%%
# We want to display the prices of each cryptocurrency
data = {
    "symbol": [],
    "price": []
}


for x in coins:
    data["symbol"].append(x["symbol"])
    data["price"].append(x["quote"]["USD"]["price"])
    print(x["symbol"], x["quote"]["USD"]["price"])

df9 = pd.DataFrame(data)
print(df9)
#%%
#Graph
"""
df_first_5 = df9.head(5)

fig = px.scatter(df_first_5, 
                 x=df_first_5.symbol, y="price", 
                 text="symbol", title="Crypto-currency price")

# Mostrar el gráfico en la consola
fig.show(renderer="svg")
"""
#%%
# Date range
today = datetime.date.today()
end = today
start = today - datetime.timedelta(weeks=52*5)

# Tickers of assets
cryptos = df9['symbol'].tolist() #We need to convert the df to a list
cryptocurrencies = [crypto + "-USD" for crypto in cryptos]

cryptocurrencies.sort()

# Downloading data
cryptodata = yf.download(cryptocurrencies, start=start, end=end) #yf needs a list before downloading data
cryptodata = cryptodata.loc[:,('Adj Close', slice(None))]
cryptodata.columns = cryptocurrencies
#%%
#%%
# Calculating returns
Y = cryptodata[cryptocurrencies].pct_change().dropna()
# %% Equal weighted Returns
w_ew = np.repeat(1/len(cryptocurrencies), len(cryptocurrencies))
w_ew = pd.DataFrame(w_ew, index=cryptocurrencies, columns=['weights'])
# %% Risk Contribution of the EW portfolio
mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets
ax = rp.plot_risk_con(w=w_ew,
                      cov=cov,
                      returns=returns,
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      erc_line = False,
                      ax=None)
plt.title("Risk (Standard Deviation) Contribution per Asset EW")
plt.show()
# %%
rp.plot_pie(w=w_ew, title='EW Portfolio', others=0.05, nrow=25, cmap="tab20c",
                 height=6, width=10, ax = None)
plt.show()
# %% Returns Distributions of the EW portfolio
ax = rp.plot_hist(returns=Y,
                  w=w_ew,
                  alpha=0.05,
                  bins=50,
                  height=6,
                  width=10,
                  ax=None)
plt.title('Portfolio Returns EW Portfolio')
plt.show()
#%% 'Calculating optimal Mean - Variance portfolio'
# Select method and estimate input parameters:
# Building the portfolio object
port_mvopt = rp.Portfolio(returns=Y)
method_mu = 'hist' # Method to estimate expected returns based on historical data.
method_cov = 'hist' # Method to estimate covariance matrix based on historical data.
port_mvopt.assets_stats(method_mu=method_mu, method_cov=method_cov)
# %%
'Estimate optimal portfolio:'
model = 'Classic'
rm = 'MV' 
obj = 'Sharpe' 
hist = True 
rf = 0
w_mvopt = port_mvopt.optimization(model=model, rm=rm, obj=obj, rf=rf, hist=hist)
# %%
'Plotting the composition of the portfolio'
rp.plot_pie(w=w_mvopt, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap="tab20c",
                 height=6, width=10, ax=None)
plt.show()
#%% Plotting the risk contribution per asset 
'Classic MV contrib'
mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets
ax = rp.plot_risk_con(w=w_mvopt,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      erc_line = False,
                      ax=None)
plt.show()
ax = rp.plot_hist(returns=Y,
                  w=w_mvopt,
                  alpha=0.05,
                  bins=50,
                  height=6,
                  width=10,
                  ax=None)
plt.title('Portfolio Returns MV Opt')
plt.show()
#%% Plotting Assets Clusters
ax = rp.plot_clusters(returns=Y,
                      codependence='spearman',
                      linkage='ward',
                      k=None,
                      max_k=10,
                      leaf_order=True,
                      dendrogram=True,
                      ax=None)
plt.show()
# %% Hierarchical Risk Clustering
# Building the portfolio object
port = rp.HCPortfolio(returns=Y)
# Estimate optimal portfolio:
model = 'HRP' # Could be HRP or HERC
codependence = 'pearson' # Correlation matrix used to group assets in clusters
rm = 'MV' # Risk measure used, this time will be variance
rf = 0 # Risk free rate
linkage = 'single' # Linkage method used to build clusters
max_k = 10 # Max number of clusters used in two difference gap statistic, only for HERC model
leaf_order = True # Consider optimal order of leafs in dendrogram
w_hrp = port.optimization(model=model,
                      codependence=codependence,
                      rm=rm,
                      rf=rf,
                      linkage=linkage,
                      max_k=max_k,
                      leaf_order=leaf_order)
w_hrp.T
# %%
# Plotting the composition of the portfolio
ax = rp.plot_pie(w=w_hrp,
                 title='HRP Naive Risk Parity',
                 others=0.05,
                 nrow=25,
                 cmap="tab20",
                 height=8,
                 width=10,
                 ax=None)
plt.show()
# %%
# Plotting the risk contribution per asset
mu = Y.mean()
cov = Y.cov() # Covariance matrix
returns = Y # Returns of the assets
ax = rp.plot_risk_con(w=w_hrp,
                      cov=cov,
                      returns=returns,
                      rm=rm,
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      erc_line = False,
                      ax=None)
plt.show()
# %% Histogram of Portfolio returns
ax = rp.plot_hist(returns=Y,
                  w=w_hrp,
                  alpha=0.05,
                  bins=50,
                  height=6,
                  width=10,
                  ax=None)
plt.title('Portfolio Returns HRC')
plt.show()