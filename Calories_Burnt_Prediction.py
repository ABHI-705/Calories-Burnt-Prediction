#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
import pickle


# In[2]:


calories=pd.read_csv("C:/Users/tripa/Downloads/archive/calories.csv")
calories


# In[3]:


# print the first 5 rows of the dataframe
calories.head()


# In[4]:


exercise_data = pd.read_csv("C:/Users/tripa/Downloads/archive/exercise.csv")

exercise_data.head()


# In[5]:


calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)
calories_data.head()


# In[6]:


# checking the number of rows and columns
calories_data.shape


# In[7]:


# getting some informations about the data
calories_data.info()


# In[8]:


# checking for missing values
calories_data.isnull().sum()


# In[9]:


# get some statistical measures about the data
calories_data.describe()


# In[10]:


sns.set()


# In[11]:


# plotting the gender column in count plot
sns.countplot(calories_data['Gender'])


# In[12]:


# finding the distribution of "Age" column
sns.distplot(calories_data['Age'])


# In[13]:


# finding the distribution of "Height" column
sns.distplot(calories_data['Height'])


# In[14]:


# finding the distribution of "Weight" column
sns.distplot(calories_data['Weight'])


# In[15]:


correlation = calories_data.corr()


# In[16]:


# constructing a heatmap to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')



# In[17]:


calories_data.replace({"Gender":{'male':0,'female':1}}, inplace=True)


# In[18]:


calories_data.head()


# In[19]:


X = calories_data.drop(columns=['User_ID','Calories'], axis=1)
Y = calories_data['Calories']

print(X)
print(Y)


# In[20]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[21]:


print(X.shape, X_train.shape, X_test.shape)


# In[22]:


# loading the model
model = XGBRegressor()


# In[23]:


# training the model with X_train
model.fit(X_train, Y_train)


# In[24]:


test_data_prediction = model.predict(X_test)
print(test_data_prediction)


# In[25]:


mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
mae


# In[26]:


filename="calories_model.sav"
pickle.dump(model, open(filename, 'wb'))


# In[27]:


loaded_model=pickle.load(open("calories_model.sav","rb"))


# In[28]:


loaded_model


# In[ ]:




