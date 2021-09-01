# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 22:05:23 2021

@author: harsh
"""

import pandas as pd 
import seaborn as sns

car_data = pd.read_csv("C:/Users/harsh/Desktop/SKILL_Edge/DataFile/T6_Luxury_Cars.csv")

car_data.isnull().sum()

sns.heatmap(car_data.isnull(),yticklabels=False, cmap="viridis")

car_data.info()

# Making dummy variable

maker_dummy = pd.get_dummies(car_data["Make"],drop_first=True)
maker_dummy.head()

type_dummy = pd.get_dummies(car_data["Type"],drop_first=True)
type_dummy.head()


origin_dummy = pd.get_dummies(car_data["Origin"],drop_first=True)
type_dummy.head()

DriveTrain = pd.get_dummies(car_data["DriveTrain"],drop_first=True)
DriveTrain.head()

car_data.isnull().sum()

# Adding  dummy data to original data set
car_data_new = pd.concat([car_data,maker_dummy,type_dummy,origin_dummy,DriveTrain],axis=1)
car_data_new.head(5)

#droping columns of whoes dummy variable is created
car_data_new.drop(["Make","Type","Origin","DriveTrain","Model"],axis=1,inplace=True)
car_data_new.head()



# taking dependent and independent variable from new dataset
x = car_data_new.drop(car_data_new.iloc[:,[3]],axis=1)
y = car_data_new["MPG (Mileage)"]

# spliting the data into test & train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)

#this is giving accurecy of 83.4%

#Coefficient
regressor.coef_

# Intercept
regressor.intercept_



#---------------------Backward Elemination-------------------------------------

import statsmodels.api as sm


import numpy as nm
X = nm.append(arr = nm.ones((426,1)).astype(int), values=x, axis=1)

x_opt=X[:,:]

regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


# Removing non significent vaiables 
x_opt=X[:, [0,2,3,4,8,16,21,24,27,37,38,41,44,45,46,47,48,51]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


# Building model after backward elemination

x_BE= X[:, [2,3,4,8,16,21,24,27,37,38,41,44,45,46,47,48,51]]
y_BE= y


from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_BE, y_BE, test_size= 0.25, random_state=0)


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_BE_train, y_BE_train)

y_pred= regressor.predict(x_BE_test)

from sklearn.metrics import r2_score
r2_score(y_BE_test,y_pred)

# accurecy = 85.7%

# after doing Backward elemination the accurecy has increased 83.4% to 85.7%

car_data_new.info()
#Coefficient
regressor.coef_

# Intercept
regressor.intercept_

'''
Mileage = 58.51 + (-0.4*cylinders) + (-0.013*horsepower) + (0.0027*weight) + 1.8*BMW + 1.22*Honda
+ 2.42*jaguar + (-1.56*Land Rover) + 3.24*MINI + 1.5*saturn  + 3.41*SCion + 2.43*Toyota + (-0.21*SUV)
+ (-0.16*Sedan) + (-0.18*Sports) + (-0.2*Truck) + (-0.17*Wagon) + 1.29*Front
'''