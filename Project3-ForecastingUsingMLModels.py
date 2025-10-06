# -*- coding: utf-8 -*-
"""
Created on Sat May  3 12:09:55 2025

@author: ASUS

#Supervised ML Algorithms
a) Linear Regression
b) Decision Tree ML
c)Random Forest ML
d) Support Vector Machine ML
e)K-Nearest Neighbour ML
f) XG Boost ML

#Theory -> Practical Quant Projects 
ML to predicts stock price using bank's  data , macro economics data


# Deceision Tree ML
o/p variable
-Continuous => Regression
-discrete => classification

eg: stock price,
    price of house,
    Interest Rate
    these are all continuous or regression
    
    
Eg of discrete:
    a) pass| Fail
    b) default| No Default
    c) Met Accident | Not met accident

    
-> when the o/p variable is 'y' is continuous in nature - Regression

# Decision Tee
-It is a supervised ML Algo
-> A Decision Tree can be used for both classification and Regression
eg: stock price using DT-- Continuous
    
     Predict whether the student will fail or pass -- Discrete

A decision tree works exactly the same way how we make decision in
real life -- by asking quesiton and replying (Yes, NO)


Example 1:
    Imagine you are trying to make a decision
 question: "should i study today or wacth netflix"
    -DO I have an exam tomorrow?
      -yes - study
      No - watch netflix
      
      
Ex 2: let say we want to predict if a student will pass or fail based on the no
of hours studied 

Yvar = Pass or Fail

Xvar = No of hours studied

#Components of decision tree
  1)Root Node
  -THis is the first question the tree ask
  .eg : is hours <= 3
  . Root node uses the most  important feature to start decision making.
 

The tree choose the question that best separates the data into pure group

how to measure purity?
Purity is measured by
a) Gini Impurity
b) Entropy | information Gain
c) Mean Square error-Regression 

#Summerize
1) Get the data(Mix)
2) Try splitting based on features
c) Measure impurity using Gini
d) Choose the split which has lowest Gini
e) Keep splitting until groups are pure

Interview Question ?
when doest the tree stop splitting
a) all sample are classified pure group
b) Max depth reached = 5
c)Min sample in a split
     The tree will not split a group if the no of samples is too small
      let says define  min_sample_splt = 4
     branch= 3 sample .. it will not split further
d) min sample in a leaf
    after a split, group -> min sample
    min_sample_leaf=2
    after splitting both sides have 2 sample.


# 

    

"""
"""
Predict BOA Stock Prict - DT Algo
* Data YF
* Data Analysis
*Feature Eng
* Train/Test
*Check how DT performance.
"""


#Predicate BAC Stock Price Using ML and Macroeconomice Variable

#Project 3:
"""
1)Build ML models to predict BAC's next day price using historical stock data,
peer financial stocks(JPM,MS,C,WFC), and key macroeconomic variable (VIX,10Y Treasury yield,
Dollar Index, Oil prices, and Gold Price)
2) Perform Feature engineering
3) Apply different ML algorithms such as Decision Tree, Random Forest, Support Vector Machine, and
    Neighbors (KNN) models
 4) Evaluate the model based on R Square, rmse, mse, mae and other metrics   
"""
#
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
#
#'BAC'= Bank of America
#'JPM' = JP Morgan Chase & Co.
#'MS'=Morgan Stanley
#'C' = Citigroup
#'WFC'= wells fargo & Co
#;SPY = S&P 500 ETF
#^VIX= CBOE Volatility Index(Market fear indication)
#'^TNX =  10Y US Tressay Yield(Interest Rate Indicator)
#'DX-Y.NYB'=> US DOller Index(Strength of US DOLLar)
#'CL=F'= #Crude Oil Future(Inflation/Energy Proxy)
#'GC=F=Gold Futures(safe assest)

#Download the data from yahoo finance
tickers=['BAC','JPM','MS','C','WFC','SPY','^VIX','^TNX','DX-Y.NYB','CL=F','GC=F']
data = yf.download(tickers,start='2002-01-01',end='2025-01-01')['Close']


data.describe()

#how many missing values ar present in my dataset
data.isnull().sum()

#Waht can we do with missing data
#1) Drop the values
#2) Forward Fill => Carries forward the last known value
#3) Backward Fill => Filling the value backward
#4) Average of the particular stock
#5) Interpolation => Linear Interpolation or Cubic Spline Interpolation or Monotone Convex Interpolation


"""
#Eg: Forward Fill
df =[100,np.nan,np.nan,200]
df= df.fillna(method='ffill') or df = df.ffill()
output:
    df=[100,100,100,200]
    
    
#Eg: Backward Fill
df =[100,np.nan,np.nan,200]
df= df.fillna(method='bfill') or df = df.bfill()
output:
    df=[100,200,200,200]
    
    
#Eg: average Fill
df =[100,np.nan,np.nan,200]

output:
    df=[100,150,150,200]

"""

#Applying forward fill on my dataset
data = data.ffill()
data.head()

data.isnull().sum()


#Correlation in our data

data.corr()

# Create a larger picture
plt.figure(figsize=(16,9))

#Plot each stock
plt.plot(data.index, data['BAC'],label='BAC:Bank of America',linewidth=2)
plt.plot(data.index, data['C'],label='C:Citigroup,linewidth=2)
plt.plot(data.index, data['JPM'],label='JPM:JP Morgna Chase & Co.',linewidth=2)
plt.plot(data.index, data['MS'],label='MS:Morgan Stanely',linewidth=2)
plt.plot(data.index, data['WFC'],label='WFC:Wells Fargo',linewidth=2)
plt.plot(data.index, data['SPY'],label='SPY:S&P 500 ETF',linewidth=2)

#Title, lable for X & Y axies
plt.title('STock Price Comparision: Marjor Banks Vs S&P500',fontsize=20)
plt.xlabel('Date',fontsize=16)
plt.ylabel('Close Price ($)', fontsize=16)

#Add Grid lines for readability
plt.grid(True, linestyle='--',alpha=0.5)
plt.legend(fontsize=12,loc='upper left')

plt.tight_layout()
plt.show()

#Feature Engineering
df = pd.DataFrame(index=data.index)

#Create lag features-  Stock Data
df ['JPM(t-1)']=data['JPM'].shift(1)
df ['BAC(t-1)']=data['BAC'].shift(1)
df ['MS(t-1)']=data['MS'].shift(1)
df ['C(t-1)']=data['C'].shift(1)
df ['WFC(t-1)']=data['WFC'].shift(1)
df ['SPY(t-1)']=data['SPY'].shift(1)

#Create Lag features - Macroeconimic data

df ['VIX(t-1)']=data['^VIX'].shift(1)
df ['10Y_Yield(t-1)']=data['^TNX'].shift(1)
df ['Gold_Futures(t-1)']=data['GC=F'].shift(1)
df ['US_Dollar_Index(t-1)']=data['DX-Y.NYB'].shift(1)
df ['Crude_Oil_Futures(t-1)']=data['CL=F'].shift(1)


#Technical Indications -5 Day Moving Averge, Rolling Volatility,
df['BAC_MA5']= data['BAC'].rolling(window=5).mean().shift(1)
df['BAC_MA10']= data['BAC'].rolling(window=10).mean().shift(1)
df['BAC_Volatility5']= data['BAC'].pct_change(5).shift(1)

#Create Target Variable
df['Target'] = data['BAC']

#Drop nana values
df = df.dropna()

# Train our ML Algo
# a) Tell what is X variables and my Y Variable - Supervised ML Algo
# b) Split our data into training and testing (90:10)
# c) Apply ML Algo
# d) Do Prediction
# e) Evaluate the model based on R2, rmse and mse
# f) Visualtion => Actual vs Forecasted


# a) Tell what is X variables and my Y Variable - Supervised ML Algo
X= df.drop('Target',axis=1)
Y=df['Target']


# b) Split our data into training and testing (90:10)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,shuffle=Flase, test_size=0.10)
                                                    
# c) Apply ML Algo: Decision Tree
from sklearn.tree import DecisionTreeRegressor
dt_model = DecisionTreeRegressor(max_depth=4) #Calling the DT Model
dt_model.fit(X_train,Y_train) #Train my DT Model

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train,Y_train)

from sklearn.neighbors import KNeighborsRegressor
knn_model=KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train,Y_train)

from sklearn.svm import SVR
svr_model =SVR()
svr_model.fit(X_train,Y_train)

# d) Do Prediction: X_test and Y_test are my actual values
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predit(X_test)
knn_pred=knn_model.predict(X_test)
svr_pred=svr_model.predict(X_test)


#Actual Vs Predicted = Decision Tree
result = pd.DataFrame(Y_test.index)
result['Actual']=Y_test.values
result['DT Prediction']=dt_pred
result['RF Prediction']=rf_pred
result['KNN Prediction'] = knn_pred
result['SVR Prediction'] = svr_pred
result
# e) Evaluate the model based on R2, rmse and mse
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(y_true,y_pred,model_name):
    r2 = r2_score(y_true,y_pred)
    mse=mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse) #rmse: root means squared error
    print("Model Name:", model_name)
    print("R2 Value",model_name)
    print("MSE", mse)
    print("RMSE",rmse)
    print("\n")
    

#Y_test = Actual Value
#dt_pred, rf_pred = Predicted Value

evaluate_model(Y_test,dt_pred,"Decision Tree")
evaluate_model(Y_test,rf_pred,"Random Forest")
evaluate_model(Y_test,knn_pred,"K Nearest Neigbhor")
evaluate_model(Y_test,svr_pred,"Support Vector Regressor")

'''
Model Name: Decision Tree
R2 Value 0.9582605359398104
MSE 1.338860276907111
RMSE 1.1570913001604977

Model Name: Random Forest
R2 Value 0.9838263249812169
MSE 0.5187965754190673
RMSE 0.7202753469466155

Model Name: K Nearest Neighbor
R2 Value -0.5904608252834846
MSE 51.01658272081046
RMSE 7.14258935686565

Model Name: Support Vector Regressor
R2 Value 0.012732827952213976
MSE 31.66817852388099
RMSE 5.627448669146702

'''


