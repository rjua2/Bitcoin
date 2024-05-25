# -*- coding: utf-8 -*-
"""
Created on Sat May 25 00:56:53 2024

@author: User
"""

import pandas as pd
import requests
import yfinance as yf


import plotly.express as px

import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import datetime

import pypfopt
from pypfopt import risk_models, expected_returns

from pypfopt import EfficientFrontier, objective_functions
from math import sqrt



from datetime import date
import datetime
global p
import time

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

# Calculate the percentage change in prices
Y = cryptodata[cryptocurrencies].pct_change().dropna()

#%%
# Define the assets
assets = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'XRP-USD']

# Define start and end dates
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=5*365)  # Last 5 years

# Download data from Yahoo Finance
data = yf.download(assets, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate expected returns and covariance matrix
mu = expected_returns.mean_historical_return(returns)
Sigma = risk_models.sample_cov(returns)

# Create the EfficientFrontier object
ef = EfficientFrontier(mu, Sigma)

# Optimize to maximize the Sharpe ratio
weights = ef.max_sharpe()

# Print the optimal portfolio weights
print("Optimal weights:", weights)

# Calculate the expected return, volatility, and Sharpe ratio of the optimized portfolio
expected_return, volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)
print("Expected Return:", expected_return)
print("Volatility:", volatility)
print("Sharpe Ratio:", sharpe_ratio)



