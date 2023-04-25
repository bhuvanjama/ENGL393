import math
import numpy as np
import pandas as pd
import yfinance as yf
import datetime
import scipy.stats as stats
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import pandas_datareader.data as web

def monteCarlo(S, K, vol, r, N, M, market_value, T, option_type):
    
    # Precompute constants
    dt = T/N
    nudt = (r - 0.5*vol**2)*dt
    volstd = vol * np.sqrt(dt)
    lnS = np.log(S)

    # Monte Carlo Method
    Z = np.random.normal(size=(N, M)) # matrix Z of size time steps by number of simulations
    
    # represent the small increments we are moving in time 
    delta_lnSt = nudt + volstd * Z    # new matrix which is the delta of the process and is drift term + vol  
    lnSt = lnS + np.cumsum(delta_lnSt, axis=0) # get cumulative sums of the deltas
    
    # concatenate numpy array of shape 1 by the number of simulations with the fill value which is the natural log of st
    # this is for completeness
    lnSt = np.concatenate((np.full(shape=(1,M), fill_value=lnS), lnSt))

    # Compute expectation and SE
    ST = np.exp(lnSt)
    if option_type == "call":
        CT = np.maximum(0, ST-K)
    elif option_type == "put":
        CT = np.maximum(0, K-ST)
    else:
        raise ValueError("Invalid option type: {}. Must be 'call' or 'put'.".format(option_type))
        
    C0 = np.exp(-r*T)*np.sum(CT[-1])/M # take the final column, take its sum, and get the discounted payoff of the average

    sigma = np.sqrt(np.sum((CT[-1]-C0)**2) / (M-1))
    SE = sigma/np.sqrt(M)
    std = np.sqrt(np.sum((CT[-1]-C0)**2) / (M-1))


    print("{} option value is ${} with SE +/- {}".format(option_type.capitalize(), np.round(C0,2), np.round(SE, 2)))

    # visualization
    x1 = np.linspace(C0-3*SE, C0-1*SE, 100)
    x2 = np.linspace(C0-1*SE, C0+1*SE, 100)
    x3 = np.linspace(C0+1*SE, C0+3*SE, 100)
    s1 = stats.norm.pdf(x1, C0, SE)
    s2 = stats.norm.pdf(x2, C0, SE)
    s3 = stats.norm.pdf(x3, C0, SE)
    plt.fill_between(x1, s1, color='tab:blue',label='> StDev')
    plt.fill_between(x2, s2, color='cornflowerblue',label='1 StDev')
    plt.fill_between(x3, s3, color='tab:blue')
    plt.plot([C0,C0],[0, max(s2)*1.1], 'k',
            label='Theoretical Value')
    plt.plot([market_value,market_value],[0, max(s2)*1.1], 'r',
            label='Market Value')
    plt.ylabel("Probability")
    plt.xlabel("Option Price")
    plt.legend()
    plt.show()

# Option Data
def options_chain(symbol):

    tk = yf.Ticker(symbol)
    # Expiration dates
    exps = tk.options

    # Get options for each expiration
    options = pd.DataFrame()
    for e in exps:
        opt = tk.option_chain(e)
        opt = pd.DataFrame().append(opt.calls).append(opt.puts)
        opt['expirationDate'] = e
        options = options.append(opt, ignore_index=True)
    
    # Boolean column if the option is a CALL
    options['CALL'] = options['contractSymbol'].str[4:].apply(
        lambda x: "C" in x)
    
    options[['bid', 'ask', 'strike']] = options[['bid', 'ask', 'strike']].apply(pd.to_numeric)
    options['mark'] = (options['bid'] + options['ask']) / 2 # Calculate the midpoint of the bid-ask
    
    # Drop unnecessary and meaningless columns
    options = options.drop(columns = ['contractSize', 'currency', 'change', 'percentChange', 
    'lastTradeDate', 'lastPrice', 'inTheMoney','openInterest','volume'])

    return options


ticker = input("Input a stock ticker: ")
option_type = input("Is this a call or put? ")
option_type = option_type.lower()
option_data = options_chain(ticker)
df = yf.download(ticker)
print(option_data.head())

S = (df.tail())['Close'][4]  # stock price
K = option_data.head()['strike'][0]  # strike price
vol = option_data.head()['impliedVolatility'][0] # implied volatility
r = 0.01 # risk free return rate
N = 10 # number of time steps
M = 1000 # number of simulations
market_value = option_data.head()['mark'][0] # market price for option

T = ((datetime.date(2023,2,24)-datetime.date.today()).days+1)/365    
monteCarlo(S, K, vol, r, N, M, market_value, T, option_type) 