# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 16:34:09 2025

@author: ASUS
"""

#Investment analyss for Equities
'''
Risk & Return Metrics Calculating for Portfolio - Theoritical

a)Alpha
It tells you how much better or worse your investment did compared to what was e
expected - after considering the risk it took.
e.g.: expected to earn 8%
in actual , your portfolio give 10%
Alpha = 10%-8% = 2%
 you earned 2% more than the expected
 
 doubt: How do people set expected return?
 Industry Benchmark = S&P 500 (USA)
                    =15% ->every year
                    
e.g.2: Risky Tech Stock Portfolio
    -your portfolio consisits of risky tech stocks
    -Given the high level of risk, your expected return was 17%
    -In Actual 10% 
    earned =10-17= 7%
     means: a - your portfolio has underperformed
            b- even though, you took more risk, still it didn't reward you enough
#Summerize on Alpha:
    a) +ve  -> you did better job by beating the index/expected return
    b) 0 -> you did same as the index/expected return
    c)-ve -> you did worse than index/expected return

b)Beta
    It measures how much your investment moves in relation to the market.
    -It is a number that tells you:
        "If the market goes up or down, how much will your portfolio move"
        
    

    
d)Returns
    ->It is the nothing but profit or loss you make froma an investment, usually
    shown as  a percentage of what you originally invested.
    e.g.
    -you buy stock ->$100
    -Next day -> $110
    Return = $110-$100/$100 = 10/100= 10%
    
    #Types of Returns
    1) Simple Return
        percentage change in an asset price overtime. It tells you how much
        your investment grew(or shrank) from one point to another
        fomula = St-St-1/St-1
      when to use? 
          ->use it when you want a quick & intuitive view of daily,weekly or
          monthly gains or losses.
      when not to use it?
          -> Avoid using simple return when you are compounding return over time.
    
        ->Is Simple Return a good measure for capturing the compounding effect
             Answer : No
          
    2)Log Return
        It is the natural logarithm of the price ratio b/w 2 time points.
        It assumes continous compounding & is widely used in Quant finance
        formula
            Log return = ln(St/St-1)  --new price/old price
            
        When to use it?
            you can use log return when working on quantitative models,
            portfolio optimization model etc
              ex: Derivative pricing
        when NOT use it?
            ->Don't  use log return when explaining return to a beginner or
            any investor
    3)Cummulative Return
        It measures the total return of an investment over a certain period
        - from start to end
        formula = Sfinal - Sinitial/Sinitial
        
        Use: TO understand how much you stock or portfolio has grown over months
        or years
        It's  basically used for long term investmnet performance
        
        when NoT to use it:
            -Daily Retunr - Don't use it
            
        
        
    4)Annulized Return
       - Formula = (1+daily return ) raised to 252 - 1 
          In a year, no of trading days is considered to tbe 252
          
          if it is log return used,
          then Annulaized -> avg dialy log return * 252
          
c)Volatility  
   -> It measures how much the price of the stock fluctuates (move up or down)        
      over time.
      -Volatility = risk measure = uncertinity in Return
      
      -> if I say that, my stock has high volatility
         it means my stock jump up & down
         
         Calulate Volatility - Formula - rt = ln(St/St-1)
            volatility = std.dev
            
      #How to Annualized this volatility
      ->Annula vol = dialy volatility * sqrt(252)
      ->Annual Return = Avg Return * 252
     Note: What volatility does not tell you?
     a) volatility is high-> fluctuation in my stok - stock price up or down
     
        
            
e)Sharpe Ratio
     ->tells you how much return you are getting for each unit risk you take.
       sharpe ratio = return per unit of risk
                    = Rp-Rf/SIGMAp  
                    Rp-> Portfolio Return
                    Rf -> Risk Free Rate
                    SIGMAp - portfolio return(Risk)
          Sharpe Ratio:
              >1 Good risk adjusted return
              >2->very good risk adj return
              >3 -> excellent
              <1 -> Riks is not well compensated

        
f)Sortino Ratio
    It measures how much return you are earning for every unit of downside risk
    very similar to sharpe ratio
    Formula = Rp-Rf/SIGMA(downside return)
    
    Share Ratio                                   Sortino
    Use total volatility(Up&DOwn)            dowside movement of stock
    
    #how to compute downside vol??
    vol -> std.dev of all return
        -> std.dev of only negative return(SIGMAd(downside risk))
    
g)Maximum Drawdown
    -measures the largest loss from a portfolio's peak to its lowest point
    -MD tells you "what is the worst drop this investment has experience during
     a certain period"
     
     Q) how dowe calculate Max drawdown
    1) Track the running maximum of the portfolio-> highest value
    2)calculated the drawdown at each point
      drawdown = current value - peak value/peak value
      
    3) The max drawdown is the most negative value in the list of drawdown
    
    Ex: Day 1=$100  
        Day 2=$110  [Peak -> highest value]
        Day 3=$105
        Day 4:$102  [Lowest point after peak]
        Day 5:$108
    
    peak = $110
    Lowest point after peak = $102
    Max Drawdown = $102-$110/$110 = -7.27%
    
    #What tell's you??
     It is basically used to understand what's  the worst drop
         
     
h)Calmar Ratio
    ->It tells you how much return you are earning for every unit of max drawdown
     Think of it as:
         "How much reward I am getting for the worst risk I have faced"
         
         Formula = Annualized Return/Max Drawdown
         
         Annualized = simple return or log return
         
    #Why does CALMAR ratio matters??
    ->Some portfolio may have high return but also massive crashes
      Calmar tells you if the risk was worth it or not
      
      Eg1 : Moderate Return & Low Max drawdown
      Annulized Return = 12%, Max Drawdown 
         
      Calmar Ratio = 12%/6% = 2 -> Strong risk Adjusted
      
      Eg2: High Return, High Risk
      Avg Annual Return = 30%
      MAx Drawdown = 25%
      CR = 30%/25% =1.2
      Note: The higher Calmar Ratio, the better it is
      
      Good Calmar Ratio > 1 the higher it is, the better it is
      Focuses -> worst loss
      Better than Sharpe Ratio-When drawdow is more important than day to day volatility
        
i)Treynor Ratio:
    -> It is risk adjusted return matric that evaluates how much excess return an
    investment earns per unit of systematic risk(Market Riks ->Beta)
    
    Formula = Rp-Rf/BETAp
    BETAp - Beta of the portfolio
    
    Eg1: Rp=12%, Rf=2%, BETAp =1
       Treynor Ratio = Rp-Rf/BETAp = 12%-2%/1 = 10% -> The portfolio earned
           10% return for each unit of market risk
           
    Conclude:
        A higher TR implies better risk adjusted performance relative to market risk
        
        Sharpe Ratio -> focuses on std.dev or Vol
        Treynor Ratio -> focuses only on Market Risk
        
    
j)ValueAtRisk( very very important Equity Portfolio)
     VaR is a risk measure that quantifies the extent of financial loss within
     a portfolio over a specific time frame at a specific confidence Interval
     
     Eg: My 1 Day 95% VaR -> $1M
     I am 95% confident that in 1 day, the extent of financial loss that can happend
     in my portfolio is $1M
     
     5% chance -> my losses can be greater than $1M
     
     VAR can be calculated using risk models ways:
         1) Variance Covariance Method
         2) Historical Method
         3)Monte Carlo Simulation method
    
    Steps to calculate VaR using Historical Method
    1) Collect stock price data
    2)Calculate Return -> simple return
    3)Arrange the return in Asecending order
    4) choose a Confidence Interval - 99% or 95%
    5) Calculate VAR
    
    100%-95% = 5% * 252 trading days = 5/100 *252 = 13.35
    whatever return you have at 13th row no is -> VaR return
    = VaR return * portfolio value
    = VaR number
    
    
     
k)Conditional Value at Risk
   also called CVaR or ES(Expected Short Fall)
   Limitation VaR: It does not tell you how bad lossed can go beyond VaR
   
   CVaR =E[Loss| Loss>VaR]
         i.e P(A|B) Prob of A given event B
         
    a)Get stock price
    b)Calculate Return
    c)Arrange in Ascending Order
    d_Choose CI ->95%
    e) alpha = 5$ Nalpha =250*5% = 12.5 ==13
    
    ES of return -> Avg of losses [1 to 12 rows] returns
    
    ES($) = ES of Return * Portfolio Value
    
   
'''

#Step 1: Fist we will try to write the code
#Setp 2: We optimize the code

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import yfinance as yf

#JPM, MS, BAC
tickers=['JPM','MS','BAC']
start_date = '2024-01-01'
end_date ='2025-01-01'
data= yf.download(tickers,start=start_date,end=end_date)['Close']

px.line(data,x=data['BAC'].index, y=data['BAC'],title='Closing Price of Bank Of America').show()
px.line(data,x=data['JPM'].index, y=data['JPM'],title='Closing Price of JP Morgan Chase',color_discrete_sequence=['green']).show()
px.line(data,x=data['MS'].index, y=data['MS'],title='Closing Price of Merrill Lync',color_discrete_sequence=['red']).show()

#Visulaize All our Stocks Data

px.line(data,title='Closing Price of prices of Bank Stocks').show()

#Calculate Simple Returns
simple_returns = data.pct_change()
simple_returns = simple_returns.dropna()

#Calculate Log Return
log_returns = np.log(data/data.shift(1))
log_returns = log_returns.dropna()

#Portfolio Returns - Rp=w1r1+w2r2+w3r3
weights = np.array([1/3,1/3,1/3])
portfolio_simple_returns = simple_returns.dot(weights)

portfolio_log_returns = log_returns.dot(weights)


#Annualize SImple Return of our Portfolio - Simple returns=(1+Avg(Returns)) raise to 252 - 1
annualized_simple_return = ((1+portfolio_simple_returns.mean())**252)-1

#Annualized Log Return of our Portfolio - Log Return = Avg Returs * 252
annualized_log_return = portfolio_log_returns.mean() * 252


#Volatility 
daily_volatility = np.std(portfolio_simple_returns)
annual_volatility = daily_volatility*np.sqrt(252)

#Calculate Alpha and Beta(SPY Data - S&P 500)
benchmark=yf.download('^GSPC',start=start_date,end=end_date)['Close']

#Calculate Simple Returns
benchmark = benchmark.pct_change()

benchmark = benchmark.dropna()

#Calculate Beta
#Beta = Covariance(Rp,Rm)/Var(Rm)
#Covariance: How 2 variable(portfolio,market) are moving together
portfolio_returns = portfolio_simple_returns.to_numpy().flatten()
benchmark_returns =benchmark.to_numpy().flatten()
cov_matrix = np.cov(portfolio_returns,benchmark_returns)

beta = cov_matrix[0,1]/cov_matrix[1,1]
#Beta = 0.90 < 1 => Our Portfolio is less volatile than the market
#For every 1% change in the market, our portfolio tend to change ~0.90%in the same direction

#Calculate Alpha
risk_free_rate = 0.07
#CAPM formula or Jensen's Alpha for calculationg ALPHA
#alpha = Rp-[Rf+BETA(Rm-Rf)]
alpha = (np.mean(portfolio_simple_returns)- risk_free_rate/252) - beta*(np.mean(benchmark_returns)-risk_free_rate/252)
alpha = alpha*252
#Results = 0.14450547716
#Our portfolio outperformed the benchmark by 14.45%

#Sortino Ratio
negative_returns = portfolio_simple_returns[portfolio_simple_returns<0]
downside_deviation = np.std(negative_returns) #Daily Downside Std Dev
downside_deviation = downside_deviation*np.sqrt(252) #Annulaized Downside Std Dev
#Sharpe Ratio = (Rp-Rf)/sigma

sharpe_ratio = (annualized_simple_return-risk_free_rate)/annual_volatility
#Results = 1.6412890
#For every 1 unit of risk, the portfolio is generating 1.64 units of excess return

##Sortino Ratio = (Rp-Rf)/sigma(d)
sortino_ratio=(annualized_simple_return-risk_free_rate)/downside_deviation
#Results = 2.586796369
#For every unit of downside risk, the portfolio is generating 2.58 units of excess returns
#Sortino Ratio of 2.58 is considerd to be really good

#Calmar Ratio =(Rp/Max Drawdown)
#Max Drawdown => Cumulative Return
cumulative_simple_returns =(1+portfolio_simple_returns).cumprod()
#0.38893656
max_drawdown=((cumulative_simple_returns.cummax() - cumulative_simple_returns)/cumulative_simple_returns.cummax()).max()
#running_max=cumulative_simple_returns.cummax()

#max_drawdown = (running_max-cumulative_simple_returns)/running_max
#Results = 0.133234516232

#Another way
'''
cumulative_simple_returns =(1+portfolio_simple_returns).cumprod()
running_max = cumulative_simple_returns.cummax()
drawdown =(running_max - cumulative_simple_returns)/running_max
drawdown.max()
'''



#Calmar Ratio
calmar_ratio = annualized_simple_return/max_drawdown
#Results 3.17322238758436

#Treynor Ratio = (Rp-Rf)/Beta

treynor_ratio =(annualized_simple_return-risk_free_rate)/beta

#Value at Risk(Historical Method)
portfolio_value =1000000
var_90= np.percentile(portfolio_simple_returns,10)* portfolio_value
var_95=np.percentile(portfolio_simple_returns,5)*portfolio_value
var_99=np.percentile(portfolio_simple_returns,1)*portfolio_value


len(portfolio_simple_returns)*10/100
len(portfolio_simple_returns)*5/100
len(portfolio_simple_returns)*1/100

#Conditional Var/Expected Shortfall=>E[Loss/Loss>VaR]
c_var=portfolio_simple_returns[portfolio_simple_returns<=np.percentile(portfolio_simple_returns,5)].mean()
c_var = c_var*portfolio_value
