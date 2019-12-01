#!/usr/bin/env python
# coding: utf-8

# ## Data preparation
# Import the dataset file into a Pandas dataframe

# In[1]:


import pandas as pd
data = pd.read_csv(r"D:\Downloads\NewDataSet.csv")
data.head()


# Evaluate whether the data needs additional cleaning

# In[2]:


data.describe()


# Convert the Pandas dataframes into numpy array that extracts only the features we am going to work with (ProductNr, Latitude, Longitude) and another array that contains the actual delivery time and an array with the names of the features

# In[3]:


x = data[['Product Nr','Latitude','Longitude']].values

y = data['Actual Delivery Time'].values

names = ['Product Nr','Latitude','Longitude']


# Splitting the data into 75% training and 25% testing

# In[28]:


import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)

# multiplying the lantitude with the longitude in order to get 1D array from 2D array
x_p = x_test[:,0:1] * x_test[:,1:2]
print(x_p)


# ## Linear Regression 

# In[30]:


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

linreg = regressor.predict(x_test)


# In[34]:


# Displaying the Linear Regression score
regressor.score(x_test, y_test)


# In[33]:


# Visualising the Linear Regression results
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call

# print("mean_absolute_error: " + str(mean_absolute_error(y_test, linreg)))
# print("mean_squared_error: " + str(mean_squared_error(y_test, linreg)))
# print("r2_score: " + str(r2_score(y_test, linreg)))
#
# plt.scatter(x_p, y_test, color="red", alpha=0.01)
# plt.scatter(x_p, linreg, color="green", alpha=0.01)
# plt.ylim(2.9, 4.2)
# plt.title("Linear Regression")
# plt.xlabel("Lantitude * Longitude")
# plt.ylabel("Delivery time")
# #plt.show()
# plt.tight_layout()
# plt.savefig("LinReg.png")


# ## Polynomial Regression

# In[35]:


#Fitting Polynomial Regression to the dataset
# from sklearn.preprocessing import PolynomialFeatures
#
# poly_reg = PolynomialFeatures(degree=4)
# X_poly = poly_reg.fit_transform(x_train)
# X_t_poly = poly_reg.fit_transform(x_test)
#
# pol_reg = LinearRegression()
# pol_reg.fit(X_poly, y_train)
#
# polynom = pol_reg.predict(poly_reg.fit_transform(x_test))
#
#
# # In[37]:
#
#
# # Visualising the Polynomial Regression results
# print("mean_absolute_error: " + str(mean_absolute_error(y_test, polynom)))
# print("mean_squared_error: " + str(mean_squared_error(y_test, polynom)))
# print("r2_score: " + str(r2_score(y_test, polynom)))
#
# plt.scatter(x_p, y_test, color='red', alpha=0.005)
# plt.scatter(x_p, polynom, color='blue', alpha=0.005)
# plt.ylim(2.9, 4.2)
# plt.title("Polynomial Regression")
# plt.xlabel("Lantitude * Longitude")
# plt.ylabel("Delivery time")
# plt.tight_layout()
# plt.savefig("PolyNom")
# plt.show()
#
#
# # ## Decision Tree Regression
#
# # In[52]:
#
#
# # # Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x_train, y_train)

decpre = regressor.predict(x_test)

print("mean_absolute_error: " + str(mean_absolute_error(y_test, decpre)))
print("mean_squared_error: " + str(mean_squared_error(y_test, decpre)))
print("r2_score: " + str(r2_score(y_test, decpre)))

plt.scatter(x_p, y_test, color="red", alpha=0.06)
plt.scatter(x_p, decpre, color="blue", alpha=0.06)
plt.ylim(0, 20)
plt.title("Decision Tree")
plt.xlabel("Long * Lat")
plt.ylabel("Delivery time")
plt.tight_layout()
plt.savefig("DecReg")
# plt.show()
#
# export_graphviz(regressor,
#                         out_file='tree.dot',
#                         feature_names=['Product Nr','Latitude','Longitude'],
#                         class_names=['Product Nr','Latitude','Longitude'],
#                         rounded=True, proportion=False,
#                         precision=2, filled=True)
#
#
# # In[55]:
#
#
# # Displaying the Decision Tree Regression score
# regressor.score(x_test, y_test)
#
#
# # In[1]:
#
#
# # Visualising the Decision Tree Regression results
#
#
# # ## Support Vector Regression
#
# # In[12]:
#
#
# # Fitting Support Vector Regression to the dataset
from sklearn.model_selection import cross_val_score
# from sklearn.svm import SVR
#
# regressor = SVR(kernel='rbf')
# regressor.fit(x_train,y_train)
#
# reg_predict = regressor.predict(x_test)
#
#
# # In[24]:
#
#
# # Visualising the Support Vector Regression results
# print("mean_absolute_error: " + str(mean_absolute_error(y_test, reg_predict)))
# print("mean_squared_error: " + str(mean_squared_error(y_test, reg_predict)))
# print("r2_score: " + str(r2_score(y_test, reg_predict)))
#
# plt.scatter(x_p, y_test, color="red", alpha=0.05)
# plt.scatter(x_p, reg_predict, color="blue", alpha=0.05)
# plt.ylim(0, 10)
# plt.title("Support Vector Regression")
# plt.xlabel("Long * Lat")
# plt.ylabel("Delivery time")
# plt.tight_layout()
# plt.savefig("SVR")
# plt.show()
#
#
# # ## Random Forest
#
# # In[19]:
#
#
# # Fitting Random forest to the dataset
# from sklearn.ensemble import RandomForestRegressor
#
# regressor = RandomForestRegressor(n_estimators=10, random_state = 1)
# regressor.fit(x_train,y_train)
#
# redict = regressor.predict(x_test)
#
#
# # In[23]:
# # i = 0
# # for estimator in regressor.estimators_:
# #     export_graphviz(estimator,
# #                         out_file=str(i) + 'tree.dot',
# #                         feature_names=['Product Nr','Latitude','Longitude'],
# #                         class_names=['Product Nr','Latitude','Longitude'],
# #                         rounded=True, proportion=False,
# #                         precision=2, filled=True)
# #     i += 1
#
#
# print("mean_absolute_error: " + str(mean_absolute_error(y_test, redict)))
# print("mean_squared_error: " + str(mean_squared_error(y_test, redict)))
# print("r2_score: " + str(r2_score(y_test, redict)))
#
# plt.scatter(x_p, y_test, color="red", alpha=0.06)
# plt.scatter(x_p, redict, color="blue", alpha=0.06)
# plt.ylim(0, 20.2)
# plt.title("Random forest")
# plt.xlabel("Long * Lat")
# plt.ylabel("Delivery time")
# plt.tight_layout()
# plt.savefig("RFR")
#plt.show()

