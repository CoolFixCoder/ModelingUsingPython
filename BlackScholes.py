# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 13:10:56 2025

@author: mubale
"""
import numpy as np
from scipy.stats import norm

#S = Stock Price
#K=Strike Price
#T = Time to Maturity
#r = risk free interest rate
#sigma = volatility of the stock
def black_scholes_call(S,K,T,r,sigma):
    #denom = 
    d1=(np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    call_option = S*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    return call_option

def black_scholes_put(S,K,T,r,sigma):
    d1=(np.log(S/K) + (r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    put_option = K*np.exp(-r*T) * norm.cdf(-d2) -S*norm.cdf(-d1)
    return put_option
    

S=100
K=95
T=3/12
r=0.04
sigma = 0.10  #(#10 percent)

print(black_scholes_call(S,K,T,r,sigma))
print(black_scholes_put(S,K,T,r,sigma))


'''
import scipy.stats as st
import numpy as np
import math

def bsformula(cp, s, k, rf, t, v, div):
        """ Price an option using the Black-Scholes model.
        cp: +1/-1 for call/put
        s: initial stock price
        k: strike price
        t: expiration time
        v: volatility
        rf: risk-free rate
        div: dividend
        """

        d1 = (np.log(s/k)+(rf-div+0.5*v*v)*t)/(v*np.sqrt(t))
        d2 = d1 - v*np.sqrt(t)

        optprice = (cp*s*np.exp(-div*t)*st.norm.cdf(cp*d1)) - (cp*k*np.exp(-rf*t)*st.norm.cdf(cp*d2))
        delta = cp*st.norm.cdf(cp*d1)
        vega  = s*np.sqrt(t)*st.norm.pdf(d1)
        return optprice, delta, vega


if __name__ == "__main__":
     ex = black_scholes(-1, 100.0, 110.0, 2.5, 0.4, 0.05, 0.0)
     
'''     
'''
from blackscholes import BlackScholesCall, BlackScholesPut

def calculate_option_prices(S, K, T, r, sigma, q):
    """
    Calculate the Black-Scholes option prices for European call and put options using the 'blackscholes' package.

    Parameters:
    S : float - current stock price
    K : float - strike price of the option
    T : float - time to maturity (in years)
    r : float - risk-free interest rate (annual as a decimal)
    sigma : float - volatility of the underlying stock (annual as a decimal)
    q : float - annual dividend yield (as a decimal)

    Returns:
    tuple - (call price, put price)
    """
    # Creating instances of BlackScholesCall and BlackScholesPut
    call_option = BlackScholesCall(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
    put_option = BlackScholesPut(S=S, K=K, T=T, r=r, sigma=sigma, q=q)

    # Get call and put prices
    call_price = call_option.price()
    put_price = put_option.price()

    return call_price, put_price


call_price, put_price = calculate_option_prices(100, 105, 1, 0.0005, 0.20, 0.0)
print("Call Price: {:.6f}, Put Price: {:.6f}".format(call_price, put_price))
'''
             
