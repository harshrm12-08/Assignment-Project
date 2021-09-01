# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 21:41:40 2021

@author: harsh
"""
#importing the libraries 
import pandas as pd
import seaborn as sns

#reading the CSV file
employeeData = pd.read_csv("C:/Users/harsh/Desktop/SKILL_Edge/DataFile/T7_HR_Data.csv")


employeeData.info()
employeeData.head()
employeeData.tail()
print("No. of employee in original dataset:" +str(len(employeeData.index)))

#Analyzing Data
sns.countplot(x="left",data=employeeData)

sns.countplot(x="left",hue="Work_accident",data=employeeData)

sns.countplot(x="left",hue="number_project",data=employeeData)

sns.countplot(x="left",hue="promotion_last_5years",data=employeeData)

#finding missing values
employeeData.isnull().sum()
sns.heatmap(employeeData.isnull(),yticklabels=False, cmap="viridis")


#creating dummy variable for role coloumn
employeeData.info()
pd.employeeData(employeeData["role"])

pd.get_dummies(employeeData["role"],drop_first=True)

role_dummy = pd.get_dummies(employeeData["role"],drop_first=True)
role_dummy.head(5)

#creating dummy variable for salary column


pd.employeeData(employeeData["salary"])

pd.get_dummies(employeeData["salary"],drop_first=True)

salary_dummy = pd.get_dummies(employeeData["salary"],drop_first=True)
salary_dummy.head(5)


#concating the data
employeeData_new = pd.concat([employeeData,role_dummy,salary_dummy],axis=1)
employeeData_new.head()

#droping role and salary column

employeeData_new.drop(["role","salary"],axis=1,inplace=True)
employeeData_new.info()


#Splitting the dataset into Train & Test dataset
x=employeeData_new.drop("left",axis=1)  #independent
y=employeeData_new["left"]    #dependent

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


predictions = logmodel.predict(X_test)


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,predictions)

#Hence, accuracy = (2130+271)/(2130+169+430+271) = 80%


#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)


#----To Improve the accuracy of the model, lets go with Backward ELimination Method &
# rebuild the logisitc model again with few independent variables--------

emp_data = employeeData_new
emp_data.head()

#--------------------------Backward Elimination--------------------------------
#Backward elimination is a feature selection technique while building a machine learning model. It is used
#to remove those features that do not have significant effect on dependent variable or prediction of output.

#Step: 1- Preation of Backward Elimination:
#Importing the library:
import statsmodels.api as sm

#Adding a column in matrix of features:
emp_data.info()    

x1=emp_data.drop("left",axis=1)  #independent
y1=emp_data["left"] #dependent


import numpy as nm
x1 = nm.append(arr = nm.ones((14999,1)).astype(int), values=x1, axis=1)


#Applying backward elimination process now
#Firstly we will create a new feature vector x_opt, which will only contain a set of 
#independent features that are significantly affecting the dependent variable.
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]


#for fitting the model, we will create a regressor_OLS object of new class OLS of statsmodels library. 
#Then we will fit it by using the fit() method.
regressor_OLS=sm.OLS(endog = y1, exog=x_opt).fit()

#We will use summary() method to get the summary table of all the variables.
regressor_OLS.summary()


#In the above summary table, we can clearly see the p-values of all the variables. 
#And remove the ind var with p-value greater than 0.05
x_opt= x1[:, [0,1,2,3,4,5,6,7,8,10,15,16,17,18]]
regressor_OLS=sm.OLS(endog = y, exog=x_opt).fit()
regressor_OLS.summary()


# leaving out RandD,hr,management,marketing,product_mng these variables because the non-significent variable

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_BE_train, x_BE_test, y_BE_train, y_BE_test= train_test_split(x_opt, y1, test_size= 0.20, random_state=0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(x_BE_train, y_BE_train)

predictions = logmodel.predict(x_BE_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_BE_test,predictions)

#Accuracy = (2128+265)/(2128+171+436+265) = 79.7%

# The accurecy has fallen down from 80% to 79.7%

#Calculating the coefficients:
print(logmodel.coef_)

#Calculating the intercept:
print(logmodel.intercept_)





















