# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:46:36 2025
Project 2:
@author: mahesh ubale
Project 2: Model Stock prices using Linear Regression and 
    test all the 5 assumption of LR Model using statistical Methods


Linear Regression is a way to find the relationship b/w 2 or more things
It's  a statistical Model
Note: It tries to draw a "straight line" through the data point

that predicts the output as cloas as possible to the real value.
->Minimise the error b/w
a) What we predicted vs what actually happen
    
when we run a LR model ,first we need to train the model
i.e. it will find a pttern->build relational b/w x&y

So 
LR is a statistical method which is used to find a relationship 
b/w a dependent variable & one or more independent variable

y = Beta+Beta1x
y-> dependent var
x->independent var

error term: captures the variation that is not explained by the model
one independent variable -> simple linear regression
more than one independent var->Multiple Linear regression

Interview question->
How do you calculate coefficient beta0, beta1
ans: The way we estimate the parameters is in such away that the 
linear regression model minimizes the sum of square errors[Residual] b/w the
Actual value & the predicted value

Assumptions of Linear Regression
->Linearity
   There should be a linear relationship b/w your dependent & independent variable
How to test:
    use VIsula Test
    if you see deviation in the straight line, drop the variable
    
    
->Independence of Error or Autocorrelation b/w residuals
  -The residual(errors) are independent of each other
  Means, that the error(difference b/w Actual &Predicated) should not follow any pattern
  If its dependent, it shows pattern and error term will be high
  if error term is high, the actual and predicted far off
  so the linear regression model will not be good.
  
  How to test: DUrbin Watson Test
  ->Durbin watson test checks for Autocorrelation in the residuals of a regression model
  ->it gives score
  ->Score is DW Test is 0 to 4
  Score       Interpretation
  2           NO Autocorrelation
  <2          positive Auto correlation
  >2          negative Auto correlation
  
  
->Homoscedasticity (constant variance)
  Th variance of residual is constant across all levels of
  independent variable
  Means: The spread of the error should be roughly the same across
  all values of i/p
  Ex: when you are predicting the salary for someone who has
  one year of work exp & someone with 20y of work experience, the error in 
  prediction should be similar
  HOw to test: 
      Breush pagan Test -Visual Test sctter plat
   if error terms is fluctuating, it become tough to predicate and make a good model
   Goal: Var should be const
   
   ->The Breush pagan Test is used to detect heteroscedasticity
   (variance of residual is not constant)
    Null Hypothesis: Residual have const variance
    Alternate Hypothesis: Residual have non-const variance
    
    p-value: Significance
    p-value <0.05 - rejct Null hypothese Ho
    p-value >0.05- always accept null hypothesis, accept alternate hypothesis
    
    In Breusn pagan Test,
     p-value > 0.05 , Model is Homoscendastic means constant variance
    
    What is NULL Hypothesis?
    baseline assumption
    Alternate Hypothesis: opposite of baseline assumption
    
    one more test is visual Test
    plot residual with predicates
    
    plot should parallel lines like funnel
     
      
->Normal Distribution of Error
  The residual should be normally distributed.
  Means: THe residual(d/f b/w Actula&predicted) should follow 
  a bell shape curve.
  Most errors should be smaall few error can be large
  How to test:
     Histogram 
     QQ Plot
     for plotting the plot of residuals
     
    Error should be normally distributed
    All your residual should be very clost to zero
Test:
    1) Visual Test:plot histogram of your residual
    2) QQ Test(Quantile Quantile Test)
      A QQ plot is a visual test for normality
      IT compares the quantile of residual to the quantile of
      a theoritical normal distribution
     observe: if it fall on a straight line then it is
     normality
     if its not falling on a straight line, it's not a
     normal distribution
     
     
     

->No Multicollinarity
  The independent var(x1,x2,x3,...) should not be highly correlated with each other
  When i/p var are hightly correlated:it makes no sense in having those 
  variable in our LR Model
  We need to make sure:i/p variable should not be closely related to each other
  This should be applicable to multiple LR model.
How to test:
    Variance Inflation Factor(VIF)
    Results
    VIF = 1->No Multicollinearity
    VIF <10 Moderate M
    VIF > 10 High M - Not acceptable
    
"""
#Quant Project 2: Linear Regression Modeling for Predicting the Stock Price and testing
#all the assumptions

#Step 1: Download the date from Yahoo Finance
#Step 2: Some Feature Engineering (to build new features) - Technical Indicators
#Step 3: Run Linear Regression Model
#Step 4: Check how the model is performed(Actual Vs Predicted)
#Step 5: Test for all the assumptions
#Step 6: Check the prediction

#Step 1:
import yfinance as yf

tickers=['AAPL','AMZN','MSFT','QQQ','^GSPC']
df = yf.download(tickers,start='2020-01-01',end='2024-12-31')['Close']

#Step 2: Perform feature Engineering
#Lesson: To Predict AAPL stock Price, we have to consider yesterday's price(all stocks)
#Th market is not open yet so we don't know what's the price today
df['AAPL(t-1)'] = df['AAPL'].shift(1)
df['AMZN(t-1)'] = df['AMZN'].shift(1)
df['MSFT(t-1)'] = df['MSFT'].shift(1)
df['QQQ(t-1)'] = df['QQQ'].shift(1)
df['^GSPC(t-1)'] = df['^GSPC'].shift(1)

#Moving Avg(MA)
df['AAPL_MA_5']=df['AAPL'].rolling(window=5).mean()
df['AMZN_MA_5']=df['AMZN'].rolling(window=5).mean()
df['MSFT_MA_5']=df['MSFT'].rolling(window=5).mean()
df['QQQ_MA_5']=df['QQQ'].rolling(window=5).mean()
df['^GSPC_MA_5']=df['^GSPC'].rolling(window=5).mean()

#Set Y Variable - Next day value
df['Target']=df['AAPL'].shift(-1)

#Set X and Y variable for Linear Regression Model - Ordinary Least Square(OLS)
X=df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']

df.dropna()
import statsmodels.api as sm

X_const = sm.add_constant(X) #Intercept Term
#Train the Model
model = sm.OLS(Y,X_const).fit()

model.summary()
#Point to remember
#If p value < 0.05 variable is signifcant -> keep the variable
#If p value > 0.05 variable is signifcant -> drop the variable
X=df[['AAPL(t-1)',  '^GSPC(t-1)', 'AAPL_MA_5']]
Y=df['Target']
X_const = sm.add_constant(X) #Intercept Term
#Train the Model
model = sm.OLS(Y,X_const).fit()

model.summary()
import pandas as pd
df_train_predict = pd.DataFrame()
df_train_predict['Actual']=df['Target']
df_train_predict['Predicted']=model.predict(X_const)
df_train_predict

import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(df_train_predict.index,df_train_predict['Actual'],label='Actual',color='black')
plt.plot(df_train_predict.index,df_train_predict['Predicted'],label='Predicted',color='red')
plt.title('Actual vs Prediction for AAPL Stock')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()

#Catch is: We need to still test LR Assumptions
#We need to check how the model is performing on test data
#Linear Regression
#Step1 : Train the model
#Step 2 : Test the model

#Assumptions of Linear Regression:
#Linearity b/w dependent and independent variable
df = df[['AAPL','AAPL(t-1)',  '^GSPC(t-1)', 'AAPL_MA_5']]

import seadborn as sns
sns.pairplot(df)
#AAPL & AAPL(t-1) has a linear relationship
#AAPL & S&P500 has a linear relationship
#AAPL & AAPL_MA_5 has a linear relationship

#2)Homoscedasticity: Fitting Residual with Predicted value
residual = model.resid #Actual - Predicted
fitted = model.fittedvalues
plt.figure(figsize=(8,5))
sns.scatter(x=fitted,y=residual)
plt.axhline(0,color='red',linestyle='--')
plt.title('Test for Homoscedasticity')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.show()
#SInce it;s a tube like structure -> it is homoscedastic->Assumption is met
#If it was funnel like structure-> it is heteroscedastic

#Multicollinearity=>VIF(Variance Inflation Factor) used for independent variables
from statsmodels.stats.outliers_influence import variance_inflation_factor
#Rule of thumb for VIF
#VIF <1 => no multicollinearity
#VIF <10=>Moderate Multicollinearity
#VIF >10 => Strong Multicollinearity

vif = pd.DataFrame()
vif['Features'] =X_const.columns
vif['VIF']=[variance_inflation_factor(X_const.values,i) for i in range(X_const.shape[i])]
vif[1:]
vif
#Results
#Features  VIF
#AAPL(t-1) 419.237303
#^GSPC(t-1) 7.635770
#APPL_MA_%  413.711580
#So drop AAPL_MA_5  from the columns
#VIF condition is met
#Again run the all tests,after removing one of the mentioned column
#so finally all the tests are met and finally good to go.
#now predict the stock price 

#4)Assumption:Normality of Residual=>1) Visual Test(Histogram) or QQ plot)
plt.figure(figsize=(6,4))
plt.hist(residual,bins=30)
plt.title("Normality of Residuals")
plt.show()

#QQ plotfor testing normality of residuals
import statsmodels.api as sm
sm.qqplot(residual,line='45',fit=True)
plt.title('QQ Plot')
plt.show()

#Step 5: Auto Correlation of Residuals; Durbing Watson Test
from statsmodels.stats.stattools import durbin_watson
dw=durbin_watson(residual)
#dw will give p-value
#P-value value<0.05 -> Autocorrelation b/w residual is there
#p-Value value>0.05-> AUtocorrelation b/w residual is not there
#0.80378765
#Our 5th condition is met


#Step 1: Download the data from Yahoo Finance predict for 3 months

tickers=['AAPL','AMZN','MSFT','QQQ','^GSPC']
df = yf.download(tickers,start='2025-01-01',end='2025-03-31')['Close']
df['AAPL(t-1)'] = df['AAPL'].shift(1)

df['^GSPC(t-1)'] = df['^GSPC'].shift(1)

df=df.dropna()
df.head()

#Prediction piece
X_test = df[['AAPL(t-1)','^GSPC(t-1)']]
X_test = sm.add_constant(X_test)

df_result = pd.DataFrame()
df_result['Actual']=df['AAPL']
df_result['Predicted'] = model.predict(X_test)

import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(df_result.index,df_result['Actual'],label='Actual',color='black')
plt.plot(df_result.index,df_result['Predicted'],label='Predicted',color='red')
plt.title('Actual vs Prediction for AAPL Stock(2025')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.tight_layout()
plt.show()

#Risk Metrics
#Calculate rmse and mse
#Rmse = Root mean square error=>(A-p)square 
#MSE = Mean square error ->(A-P) square it and take mean of it
from sklearn.metrics import mean_squared_error
#Calculate mse
mse = mean_squared_error(df_result['Actual'], df_result['Predicted'])
rmse = np.sqrt(mse)
print(rmse,mse)
#4.165605174338791 17.352266468561417
#RMSE -4.16-> The predicted stock price are, on average, off by $4 
#from the actual price



#Conclusion: It's a decent Model but not 100% Accurate
#Lesson: Stock data in general have lof of non linearlites
#It's extremely tought to use simple linear regression model just ot capture the non linear affect
#That's why in the industry it's common to use ML Models which are 
#great in capturing Non Linearities

'''
############LASSO REGRESSION  ##############
Lasso Regression stands
L->Least
A->Absolute
S->Shrinkage And
S->Selection
O->Operator

->LASSO is a type of LR model that adds a penalty to the model

LASSO Regression says:
     if the variables aren't contributing, i will make it zero
How does Lasso Reg makes the coefficient makes zero?
by adding a penality term

#OBjective fn or cost fn in Lasso Regression
cost fnof Linear Regression => summation(from i=1 to n)(yi-yi(hat))square
    yi = actual
    yi(hat) = predictal value
    
how to minimize this term? by changing coeff of the var

cost fn of Lasso Regression
cost fn(Lasso) = summation(i=0 to n)(yi-yiHat)square + 
                 lamda * summation(j=1 to p) moduls(BETAj)
              so Linear Regression + Penality Term
              
              
              
yi= acutal value
yihat=predicted value
lamda = regularization parameter-> controls the strength of the penalty
BETAj = coeff of the model
p= no of features

#When lamda = 0, Lasso Regression become OLS
As lamda increases, some coefficient will become smaller
                    some coeff => exactly zero
                    
-> Lasso Regression helps to reduce model complexity.It automatically 
  reduces features.

#Lasso Regularization is also called L1 Regularization.


'''

###########  LASSO REGRESSION   ######################

#Step 1: Download the date from Yahoo Finance

tickers=['AAPL','AMZN','MSFT','QQQ','^GSPC']
df = yf.download(tickers,start='2020-01-01',end='2025-04-01')['Close']

#Step 2: Some Feature Engineering (to build new features) - Technical Indicators

df['AAPL(t-1)'] = df['AAPL'].shift(1)
df['AMZN(t-1)'] = df['AMZN'].shift(1)
df['MSFT(t-1)'] = df['MSFT'].shift(1)
df['QQQ(t-1)'] = df['QQQ'].shift(1)
df['^GSPC(t-1)'] = df['^GSPC'].shift(1)

#Moving Avg(MA)
df['AAPL_MA_5']=df['AAPL'].rolling(window=5).mean()
df['AMZN_MA_5']=df['AMZN'].rolling(window=5).mean()
df['MSFT_MA_5']=df['MSFT'].rolling(window=5).mean()
df['QQQ_MA_5']=df['QQQ'].rolling(window=5).mean()
df['^GSPC_MA_5']=df['^GSPC'].rolling(window=5).mean()

#Set Y Variable - Next day value
df['Target']=df['AAPL'].shift(-1)
df.dropna()

#Set X and Y variable for Linear Regression Model - Ordinary Least Square(OLS)
X=df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']


#Step 1: Import all the required libraries
#Step 2: Define Features and Target Variables
#Step 3: Train Test Split
#Step 4: Apply Lasso Regression
#Step 5: Get Intercept and Coeff for Lasso Regression
#Step 6: Predict using Lasso Regression
#Step 7: Create a dataframe with Actual and Predicted Values
#Step 8: Plot Actual & Predicted Values
#Step 9: Evaluate the Model - R Square, mse, rmse


#Step 1: Import all the required libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklear.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


#Step 2: Define Features and Target Variables

X= df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']


#Step 3: Train Test Split

X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.044, shuffle=False)


#Step 4: Apply Lasso Regression

lasso= Lasso(alpha=0.1)
lasso.fit(X_train,Y_train) # Training the model


#Step 5: Get Intercept and Coeff for Lasso Regression
coefficients= lasso.coef_
coefficients
#Result are
'''
array([0.58112307,0.00318715,0.04167649,0.0053153,0.00197498,
       0.40051673,-0.,-0.03055606,-0.00437146,-0.00231246 ])
'''
intercept = lasso.intercept_
#0.5631234035895147
coeff_df = pd.DataFrame({'Feature':X.colums,'Coefficients':coefficients})

#Results
'''
 Feature  Coefficients
 AAPL(t-1)     0.58112307,
AMZN(t-1)        0.00318715,
MSFT(t-1)         0.04167649,
QQQ(t-1)         0.0053153,
 ^GSPC(t-1)        0.00197498,
AAPL_MA_5         0.40051673,
AMZN_MA_5         -0.,
 MSFT_MA_5        -0.03055606,
 QQQ_MA_5        -0.00437146,
 ^GSPC_MA_5        -0.00231246
         
'''


#Step 6: Predict using Lasso Regression
y_pred = lasso.predict(X_test)


#Step 7: Create a dataframe with Actual and Predicted Values
df_result = pd.DataFrame({'Actual:'y_test,'Predicted':y_pred})

#Step 8: Plot Actual & Predicted Values


plt.figure(figsize=(14,6))
plt.plot(df_result.index,df_result['Actual'],label='Actual',color='black')
plt.plot(df_result.index,df_result['Predicted'],label='Predicted',color='red')
plt.title('Actual vs Prediction for AAPL Stock(2025)- Lasso Regression')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.tight_layout()
plt.show()

#Step 9: Evaluate the Model - R Square, mse, rmse#Step 2: Perform feature Engineering
r2=r2_score(Y_test,y_pred)
print(" R square:":r2)

mse = mean_squared_error(Y_test,y_pred)
print("MSE :",mse)

rmse = np.sqrt(mse)
print("RMSE : ",rmse)

#Result
'''
R Square 0.7525519818994182
mse = 35.002163623249146
rmse = 5.916262639813174
'''

#Lesson: To Predict AAPL stock Price, we have to consider yesterday's price(all stocks)
#Th market is not open yet so we don't know what's the price today

############### Step 3: RIDGE REGRESSION ###################
#Step 1: Import all the required libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklear.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


#Step 2: Define Features and Target Variables

X= df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']


#Step 3: Train Test Split

X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.044, shuffle=False)


#Step 4: Apply Ridge Regression

ridge= Rigde(alpha=10)
ridge.fit(X_train,Y_train) # Training the model


#Step 5: Get Intercept and Coeff for Ridge Regression
coefficients= ridge.coef_
coefficients
#Result are
'''
array([0.58112307,0.00318715,0.04167649,0.0053153,0.00197498,
       0.40051673,-0.,-0.03055606,-0.00437146,-0.00231246 ])
'''
intercept = ridge.intercept_
#0.5631234035895147
coeff_df = pd.DataFrame({'Feature':X.colums,'Coefficients':coefficients})

#Results
'''
 Feature  Coefficients
 AAPL(t-1)     0.572550,
AMZN(t-1)      -0.000000,
MSFT(t-1)         0.005933,
QQQ(t-1)         0.000000,
 ^GSPC(t-1)        0.002456,
AAPL_MA_5         0.367765,
AMZN_MA_5         -0.000000,
 MSFT_MA_5        0.000000,
 QQQ_MA_5        0.0000000,
 ^GSPC_MA_5        0.000000
         
'''


#Step 6: Predict using Lasso Regression
y_pred = ridge.predict(X_test)


#Step 7: Create a dataframe with Actual and Predicted Values
df_result = pd.DataFrame({'Actual:'y_test,'Predicted':y_pred})

#Step 8: Plot Actual & Predicted Values


plt.figure(figsize=(14,6))
plt.plot(df_result.index,df_result['Actual'],label='Actual',color='black')
plt.plot(df_result.index,df_result['Predicted'],label='Predicted',color='red')
plt.title('Actual vs Prediction for AAPL Stock(2025)- Ridge Regression')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.tight_layout()
plt.show()

#Step 9: Evaluate the Model - R Square, mse, rmse#Step 2: Perform feature Engineering
r2=r2_score(Y_test,y_pred)
print(" R square:":r2)

mse = mean_squared_error(Y_test,y_pred)
print("MSE :",mse)

rmse = np.sqrt(mse)
print("RMSE : ",rmse)

#Result
'''
R Square 0.7443734529688026
mse = 36.15903774179844
rmse = 6.013238540237569
'''

############ ELASTIC Net - Lasso + Ridge   ######
#Step 1: Download the date from Yahoo Finance

tickers=['AAPL','AMZN','MSFT','QQQ','^GSPC']
df = yf.download(tickers,start='2020-01-01',end='2025-04-01')['Close']

#Step 2: Some Feature Engineering (to build new features) - Technical Indicators

df['AAPL(t-1)'] = df['AAPL'].shift(1)
df['AMZN(t-1)'] = df['AMZN'].shift(1)
df['MSFT(t-1)'] = df['MSFT'].shift(1)
df['QQQ(t-1)'] = df['QQQ'].shift(1)
df['^GSPC(t-1)'] = df['^GSPC'].shift(1)

#Moving Avg(MA)
df['AAPL_MA_5']=df['AAPL'].rolling(window=5).mean()
df['AMZN_MA_5']=df['AMZN'].rolling(window=5).mean()
df['MSFT_MA_5']=df['MSFT'].rolling(window=5).mean()
df['QQQ_MA_5']=df['QQQ'].rolling(window=5).mean()
df['^GSPC_MA_5']=df['^GSPC'].rolling(window=5).mean()

#Set Y Variable - Next day value
df['Target']=df['AAPL'].shift(-1)
df.dropna()

#Set X and Y variable for Linear Regression Model - Ordinary Least Square(OLS)
X=df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']


#Step 1: Import all the required libraries
#Step 2: Define Features and Target Variables
#Step 3: Train Test Split
#Step 4: Apply Lasso Regression
#Step 5: Get Intercept and Coeff for Lasso Regression
#Step 6: Predict using Lasso Regression
#Step 7: Create a dataframe with Actual and Predicted Values
#Step 8: Plot Actual & Predicted Values
#Step 9: Evaluate the Model - R Square, mse, rmse


#Step 1: Import all the required libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklear.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


#Step 2: Define Features and Target Variables

X= df[['AAPL(t-1)', 'AMZN(t-1)',
       'MSFT(t-1)', 'QQQ(t-1)', '^GSPC(t-1)', 'AAPL_MA_5', 'AMZN_MA_5',
       'MSFT_MA_5', 'QQQ_MA_5', '^GSPC_MA_5']]

Y=df['Target']


#Step 3: Train Test Split

X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.044, shuffle=False)


#Step 4: Apply ElasticNet Regression

elastic_net= ElasticNet(alpha=1,l1_ratio=0.5)
#alpha control the strength of regularization(higher alpha = stronger penalty) - Lamba parameter
#l1_ratio=0.5 => applying 50% lasso and 50% as ridge regression - alpha parameter(theory)



elastic_net.fit(X_train,Y_train) # Training the model


#Step 5: Get Intercept and Coeff for ElasticNet Regression
coefficients= elastic_net.coef_
coefficients
#Result are
'''
array([0.58112307,0.00318715,0.04167649,0.0053153,0.00197498,
       0.40051673,-0.,-0.03055606,-0.00437146,-0.00231246 ])
'''
intercept = elastic_net.intercept_
#0.5631234035895147
coeff_df = pd.DataFrame({'Feature':X.colums,'Coefficients':coefficients})

#Results
'''
 Feature  Coefficients
 AAPL(t-1)     0.58112307,
AMZN(t-1)        0.00318715,
MSFT(t-1)         0.04167649,
QQQ(t-1)         0.0053153,
 ^GSPC(t-1)        0.00197498,
AAPL_MA_5         0.40051673,
AMZN_MA_5         -0.,
 MSFT_MA_5        -0.03055606,
 QQQ_MA_5        -0.00437146,
 ^GSPC_MA_5        -0.00231246
         
'''


#Step 6: Predict using Lasso Regression
y_pred = elastic_net.predict(X_test)


#Step 7: Create a dataframe with Actual and Predicted Values
df_result = pd.DataFrame({'Actual:'y_test,'Predicted':y_pred})

#Step 8: Plot Actual & Predicted Values


plt.figure(figsize=(14,6))
plt.plot(df_result.index,df_result['Actual'],label='Actual',color='black')
plt.plot(df_result.index,df_result['Predicted'],label='Predicted',color='red')
plt.title('Actual vs Prediction for AAPL Stock(2025)- ElasticNet Regression')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.grid(True)
plt.tight_layout()
plt.show()

#Step 9: Evaluate the Model - R Square, mse, rmse

r2=r2_score(Y_test,y_pred)
print(" R square:":r2)

mse = mean_squared_error(Y_test,y_pred)
print("MSE :",mse)

rmse = np.sqrt(mse)
print("RMSE : ",rmse)

#Result
'''
R Square 0.7513311053139469
mse = 35.17485978115933
rmse = 5.930839719732723
'''


##### Performance for All Our Models
#OLS
r-square = 0.993
mse=17.35226637043703
rmse = 4.165605162570864
#LASSO 
r-square = 0.674574742009648297
mse= 33.88523829601259
rmse = 5.82110284190312
#RIDGE
R-square = 0.655523488986780858
mse= 35.898959606393746
rmse = 5.9915740508145054
#ElasticNet
r-square = 0.6638572193358945
mse = 35.0011531294005
rmse = 5.9161772395188175

