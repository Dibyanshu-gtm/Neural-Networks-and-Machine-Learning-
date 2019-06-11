#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


from sklearn.datasets import load_boston


# In[4]:


bst_data=load_boston()


# In[5]:


import matplotlib.pyplot as plt


# In[6]:


import numpy as np
import seaborn as sns


# In[7]:


boston = pd.DataFrame(bst_data.data, columns=bst_data.feature_names)
boston.head()


# In[8]:


boston['MEDV']=bst_data.target


# In[9]:


boston.head()


# In[10]:


boston.isnull().sum()


# In[11]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.distplot(boston['MEDV'],bins=30)
plt.show()


# In[12]:


corr_mat=boston.corr().round(2)


# In[13]:


sns.heatmap(data=corr_mat,annot=True)


# In[14]:


plt.figure(figsize=(20,5))


# In[17]:


plt.figure(figsize=(20,5))
features=['LSTAT','RM']
target=boston['MEDV']

for i,col in enumerate(features):
    plt.subplot(1,len(features),i+1)
    x=boston[col]
    y=target
    plt.scatter(x,y,marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')


# In[18]:


X=pd.DataFrame(np.c_[boston['LSTAT'],boston['RM']],columns=['LSTAT','RM'])
Y=boston['MEDV']


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=20,random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[23]:


# model evaluation for training set
from sklearn.metrics import r2_score
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))


# In[ ]:




