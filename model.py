#!/usr/bin/env python
# coding: utf-8

# # Uploading the required data

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import random
from sklearn.ensemble import GradientBoostingClassifier
random.seed(0)


# # Uploading the data

# In[2]:


df = pd.read_csv('deploy_data.csv')


# In[3]:


df.drop(['Unnamed: 0'], axis=1, inplace = True)


# Removing useless data

# In[4]:


df.head()


# In[5]:


df.iloc[200]


# In[6]:


columns = [col for col in df.columns]

# Converting column types of .astype in a for loop
for col in columns:
  
  df[col] = pd.to_numeric(df[col], errors='coerce')
  df[col] = df[col].astype(float)


# In[7]:


df.head(2)


# In[8]:


df.info()


# # GBT Classifier

# Data Preprocessing will be done with the help of following script lines.
# 
# 

# In[9]:


# Putting feature variable to X
X = df.drop('skipped',axis=1)
# Putting response variable to y
y = df['skipped']


# In[10]:


X = pd.get_dummies(X)


# Next, we will divide the data into train and test split. Following code will split the dataset into 90% training data and 10% of testing data −

# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 30)


# Next, data scaling will be done as follows −
# 
# 

# In[12]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# Next, train the model with the help of GBT class of sklearn as follows −
# 
# 

# In[13]:


#Import GBT model
from sklearn.ensemble import GradientBoostingClassifier

#Create a GBT Classifier
gradient_booster = GradientBoostingClassifier(learning_rate=0.1)

#Train the model using the training sets
gradient_booster.fit(X_train, y_train)


# At last we need to make prediction. It can be done with the help of following script −
# 
# 

# In[14]:


y_pred = gradient_booster.predict(X_test)


# Next, print the results as follows −
# 
# 

# In[15]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[16]:


y_pred


# In[17]:


y_test


# In[18]:


df.columns


# In[19]:


import pickle
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(gradient_booster, open('model.plk', 'wb'))

