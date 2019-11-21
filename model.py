#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Implement simple classification model on pima insians dataset.


# In[ ]:


#----Read_dataset_from_csv_file....


# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset=pd.read_csv('file:///F:/ML/diabatic/diabetes.csv')


# In[2]:


dataset.head()


# In[3]:


#print dataset['Outcome'].value_counts()


# In[20]:





# In[4]:



import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

ax = sns.countplot(x="Outcome", data=dataset)
plt.show()


# In[28]:


x=dataset['Age']
y=dataset['Pregnancies']


# In[30]:


sns.regplot(x,y)
plt.show()


# In[5]:


import numpy as np
np.mean(dataset['Age'])


# In[35]:


dataset.corr()


# In[36]:


#-----Implement_preprocessing


# In[40]:


#------find_missing_values

#print dataset.isnull().sum()
#print dataset.isnull().any()
#print dataset.columns[dataset.isnull().any()]


# In[41]:


#--------------Split_dataset_into_train_data_and_test_data---------


# In[6]:


from sklearn.model_selection import train_test_split
target=dataset['Outcome']
dataset=dataset.drop('Outcome',axis=1)


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(dataset,target,test_size=0.20,random_state=42)


# In[8]:


X_train.shape,y_train.shape,X_test.shape,y_test.shape


# In[9]:


#----Define_model--------

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50)

#----Implement_model---------

model.fit(X_train,y_train)
model


# In[10]:


#-----Predict_test_data--------

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[6,148,72,35,0,33.6,0.627,50]]))


# In[54]:





# In[2]:





# In[ ]:




