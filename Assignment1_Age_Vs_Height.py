# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 21:21:18 2021

@author: harsh
"""

#Explore the relationship between Age and Height

import pandas as pd

#dataframe = pd.read_excel('C:/Users/harsh/Desktop/SKILL_Edge/DataFile/Exercise1.xlsx')

data = {'Age':pd.Series([18,19,20,21,22,23,24,25,26,27,28,29]),
        'Height':pd.Series([76.1,77,78.1,78.2,78.8,79.7,79.9,81.1,81.2,81.8,82.8,83.5])}

dataframe1 = pd.DataFrame(data)
print(dataframe1)
#Q1 check that age and height have the same number of element

dataframe1.shape

print(len(dataframe1['Age']),' ',len(dataframe1['Height']))

dataframe1.size

#Q2 Determine the relationship between age and height

import matplotlib.pyplot as plt

plt.scatter(dataframe1['Age'],dataframe1['Height'],c='b',marker='*')
plt.xlabel('Age', fontsize=16)
plt.ylabel('Height', fontsize=16)
plt.title('scatter plot - Age vs Height',fontsize=20)


#Q3 Computing a "Linear model" that fits the data given:-

X = dataframe1.iloc[:,:-1].values
Y = dataframe1.iloc[:, 1].values

print(X)
# Splitting the dataframe into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 0)


# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)


#Calculating the coefficients:
print(regressor.coef_)

#Calculating the intercept:
print(regressor.intercept_)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)
Y_pred
dataframe1

from sklearn.metrics import r2_score
r2_score(Y_test,Y_pred)


#Q4 Determine the equation of the line of best fit

#Create a DataFrame
new_data = {'Actual Age':Y_test,'Predicted Age':Y_pred}
df = pd.DataFrame(new_data,columns=['Actual Age','Predicted Age'])
print(df)

# Visualising the predicted results
line_chart1 = plt.plot(X_test,Y_pred, '--', c ='red') # actual age vs predicted height
line_chart2 = plt.plot(X_test,Y_test, '*', c='blue') # actual age vs actual height
print(X_test,'  ',Y_test)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Height', fontsize=16)


















