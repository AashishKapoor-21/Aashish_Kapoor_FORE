#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix


# In[2]:


#importing datasets
data = pd.read_csv('/Users/aashish/Desktop/session3_datafile_mlp.csv')
#to view top n rows of our dataset
data.head(6)


# In[3]:


data.info()


# In[4]:


status = data.Loan_Status.value_counts()
print(status)


# In[5]:


plt.pie(status,labels = ("YES", "NO"),autopct = "%.2f%%")
plt.show()


# In[6]:


myexplode = [0.2, 0]
plt.pie(status,labels = ("YES", "NO"),autopct = "%.2f%%",startangle = 90,explode = myexplode,shadow = True)
plt.legend(title = "LOAN STATUS")
plt.show()


# In[7]:


data.isna().sum()


# In[8]:


new_data = data.dropna()
x= new_data.isna().sum()
y=data.isna().sum()
print(x,"\n\n",y)


# In[9]:


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


# In[10]:


data['Gender'] = data['Gender'].replace({'Male':0, 'Female':1,'unknown' : 2})
data['Married'] = data['Married'].replace({'Yes' :1, 'No': 0, 'unknown':2})
data['Education'] = data['Education'].replace ({'Graduate' : 1, 'Not Graduate' : 0})
data['Self_Employed'] = data['Self_Employed'].replace ({'Yes': 1,'No' : 0, 'unknown':2})
data['Dependents'] = data['Dependents'].replace({'3+':3})
data['Property_Area'] = data['Property_Area'].replace ({'Semiurban': 1,'Urban' : 0, 'Rural':2})
data['Loan_Status'] = data['Loan_Status'].replace({'Y':1, 'N':0})
data.head()


# In[20]:


data.dropna(inplace=True, axis=0)
X = data.iloc[:,1:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = tts(X, y, test_size = 0.3, random_state=100)
X.head() 


# In[21]:


from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
LR = KNeighborsClassifier(n_neighbors=10)


# In[22]:


#fiting the model
LR.fit(X_train, y_train)


# In[23]:


#prediction
y_pred = LR.predict(X_test)


# In[24]:


#Accuracy
print("Accuracy ", LR.score(X_test, y_test)*100)


# In[32]:


Gen= input("Input Gender 1 for Male 0 for Female: ")
Marr= input("If marrried Input 1 for Yes and 0 for No: ")
Depen= input("Depedents present ? \n Input 1, 2 or 3+ in case of more than 3: ")
Edu= input ("Education level \n Input 0 for Not Graduate 0 and 1 for Graduate: ")
SelfEmp= input("Self employed ? \nInput 1 for Yes 0 for No: ")
AppInc= input("Enter Applicant income: ")
CoApInc=input("Enter co Applicant income: ")
LoAmt=input("Enter loan amount: ")
LoAmtTerm=input("Enter loan amount term: ")
Crehis=input("Enter credit history: ")
PropAre=input("Enter property area1 for urban and 0 for rural: ")
X_actual_values=[Gen,Marr,Depen,Edu,SelfEmp,AppInc,CoApInc,LoAmt,LoAmtTerm,Crehis,PropAre]
X_actual_values


# In[33]:


X_actual_values=np.array(X_actual_values).astype('int16')
X_actual_values=X_actual_values.reshape(1,11)
X_actual_values=pd.DataFrame(X_actual_values)
X_actual_values.columns=(['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
y_actual_pred=LR.predict(X_actual_values)
print('Should the person be given a loan ? \n1 for yes 0 for no. \n As per KNN the answer is =',y_actual_pred)


# In[ ]:




