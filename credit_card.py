#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[17]:


dataset = pd.read_csv('creditcard.csv')


# In[18]:


dataset.shape


# In[19]:


dataset.isna().sum()


# In[20]:


dataset.head()


# In[21]:


pd.value_counts(dataset['Class'])


# In[22]:


sns.countplot(dataset['Class'])


# In[23]:


corrmat = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corrmat , vmax=0.8 , square=True)
plt.show()


# In[24]:


len(dataset[dataset['Class']==0]) #valid transaction


# In[25]:


len(dataset[dataset['Class']==1]) #fradulent transactions


# In[26]:


X = dataset.iloc[: , :-1].values
y = dataset.iloc[: , -1].values


# In[27]:


pip install -U imbalanced-learn


# In[29]:


pip install imbalanced-learn


# In[30]:


#convert imbalanced data to balanced data
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
x_res , y_res = ros.fit_resample(X,y)


# In[31]:


X.shape


# In[32]:


x_res.shape


# In[33]:


from collections import Counter
print(Counter(y))
print(Counter(y_res))


# In[34]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x_res , y_res , test_size=0.3 , random_state=42)


# In[35]:


x_train.shape


# In[36]:


y_train.shape


# In[54]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 250 , random_state=0, n_jobs=-1)
classifier.fit(x_train , y_train)


# In[55]:


y_pred = classifier.predict(x_test)


# In[56]:


n_errors = (y_pred != y_test).sum()


# In[57]:


n_errors


# In[58]:


y_test.shape


# In[59]:


from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(y_test , y_pred)
sns.heatmap(cm , annot=True)
print(accuracy_score(y_test , y_pred))


# In[60]:


from sklearn.metrics import precision_score
precision_score(y_test , y_pred)


# In[61]:


from sklearn.metrics import recall_score
recall_score(y_test , y_pred)


# In[62]:


from sklearn.metrics import classification_report
print(classification_report(y_test , y_pred))


# In[ ]:




