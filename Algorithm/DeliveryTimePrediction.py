#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

data = pd.read_csv('DeliveryTime.csv')
data.head()


# In[2]:


data.describe()


# In[3]:


x = data[['Product Nr','Latitude','Longitude']].values


y = data['Actual Delivery Time'].values

names = ['Product Nr','Latitude','Longitude']


# In[4]:


import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1000)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.75, random_state=1)


# In[5]:


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[6]:


y_pred = regressor.predict(x_test)


# In[7]:


regressor.score(x_test, y_test)


# In[8]:


print(regressor.coef_)


# In[41]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(x_test)
pol_reg = LinearRegression()
pol_reg.fit(x_poly, y)


# In[42]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import export_graphviz
from IPython.display import Image
from subprocess import call


print("mean_absolute_error: " + str(mean_absolute_error(y_test, polynom)))
print("mean_squared_error: " + str(mean_squared_error(y_test, polynom)))
print("r2_score: " + str(r2_score(y_test, polynom)))
plt.scatter(x_test, y_test, color='red')
plt.plot(x_test, y_test, color='blue')
plt.title("Set Title")
plt.xlabel("Set Label")
plt.ylabel("Set Label")
plt.show()


# In[ ]:




