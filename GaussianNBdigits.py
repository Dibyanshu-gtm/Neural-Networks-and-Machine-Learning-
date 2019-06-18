#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
ld_data=load_digits()


# In[2]:


ld_data.data.shape


# In[3]:


fig=plt.figure(figsize=(6,6))


# In[4]:


fig=plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0, right=1,bottom=0,top=1,hspace=0.05, wspace=0.05)
for i in range(64):
    ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(ld_data.images[i],cmap=plt.cm.Greens, interpolation="nearest")
    ax.text(0,7,str(ld_data.target[i]))


# In[5]:


plt.figure()
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
proj=pca.fit_transform(ld_data.data)
plt.scatter(proj[:, 0], proj[:, 1], c=ld_data.target, cmap="Paired")
plt.colorbar()


# In[7]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(ld_data.data, ld_data.target)

# train the model
clf = GaussianNB()
clf.fit(X_train, y_train)

# use the model to predict the labels of the test data
predicted = clf.predict(X_test)
expected = y_test

# Plot the prediction
fig = plt.figure(figsize=(6, 6))  # figure size in inches
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# plot the digits: each image is 8x8 pixels
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary,
              interpolation='nearest')

    # label the image with the target value
    if predicted[i] == expected[i]:
        ax.text(0, 7, str(predicted[i]), color='green')
    else:
        ax.text(0, 7, str(predicted[i]), color='red')


# In[8]:


matches=(predicted==expected)
matches.sum()


# In[10]:


len(matches)


# In[14]:


matches.sum()/(float(len(matches)))


# In[16]:


from sklearn import metrics
print(metrics.classification_report(expected,predicted))


# In[17]:


print(metrics.confusion_matrix(expected, predicted))

plt.show()


# In[ ]:




