#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix


# In[5]:


#importing datasets
data = pd.read_csv('/Users/aashish/Desktop/session3_datafile_mlp.csv')
#to view top n rows of our dataset
data.head(6)


# In[6]:


data.info()


# In[13]:


status = data.Loan_Status.value_counts()
print(status)


# In[23]:


plt.pie(status,labels = ("YES", "NO"),autopct = "%.2f%%")
plt.show()


# In[21]:


myexplode = [0.2, 0]
plt.pie(status,labels = ("YES", "NO"),autopct = "%.2f%%",startangle = 90,explode = myexplode,shadow = True)
plt.legend(title = "LOAN STATUS")
plt.show()


# In[25]:


data.isna().sum()


# In[43]:


new_data = data.dropna()
x= new_data.isna().sum()
y=data.isna().sum()
print(x,"\n\n",y)


# In[77]:


#converting into numerical data
#Yes: 1
# No : 0
# unknown:2
data['Gender_n'] = data['Gender'].replace({'Male':0, 'Female':1,'unknown' : 2})
data['Married_n'] = data['Married'].replace({'Yes' :1, 'No': 0, 'unknown':2})
data['Education_n'] = data['Education'].replace ({'Graduate' : 1, 'Not Graduate' : 0})
data['Self_Employed_n'] = data['Self_Employed'].replace ({'Yes': 1,'No' : 0, 'unknown':2})
data['Dependents'] = data['Dependents'].replace({'3+':3})
data.head()





# In[90]:


data['Gender'] = data['Gender'].replace({'Male':0, 'Female':1,'unknown' : 2})
data['Married'] = data['Married'].replace({'Yes' :1, 'No': 0, 'unknown':2})
data['Education'] = data['Education'].replace ({'Graduate' : 1, 'Not Graduate' : 0})
data['Self_Employed'] = data['Self_Employed'].replace ({'Yes': 1,'No' : 0, 'unknown':2})
data['Dependents'] = data['Dependents'].replace({'3+':3})
data['Property_Area'] = data['Property_Area'].replace ({'Rural': 1,'Urban' : 0, 'Semiurban':2})
data['Loan_Status'] = data['Loan_Status'].replace({'Y':1, 'N':0})
data.head()


# In[91]:


X = new_data.iloc[:,1:-1]
Y = new_data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size = 0.3, random_state=100)
X.head()


# In[92]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

LR = KNeighborsClassifier(n_neighbors=10)


# In[93]:


#fiting the model
LR.fit(X_train, Y_train)


# In[ ]:




