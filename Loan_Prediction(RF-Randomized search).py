#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ## DATA PREPROCESSING

# In[2]:


data = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/Loan_Prediction/train.csv")


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.dtypes


# In[6]:


data.columns


# In[7]:


data.shape


# In[8]:


data.isna().sum()


# In[9]:


data.isna().sum()[data.isna().sum()!=0]*100/data.shape[0]


# In[10]:


#missing object columns

missingobj = data[data.dtypes[data.dtypes==object].index].isna().sum()[(data.isna().sum()!=0)].index
missingobj


# In[11]:


#missing float columns

missingfloat=data[data.dtypes[data.dtypes=='float64'].index].isna().sum()[(data.isna().sum()!=0)].index
missingfloat


# In[12]:


#filling missing values of object columns

for x in missingobj:
    data[x][data[x].isna()] = data[x].mode()[0]


# In[13]:


#filling missing values of float64 columns

for x in missingfloat:
    data[x][data[x].isna()] = data[x].mean()


# In[14]:


data.isna().sum()


# In[15]:


databackup = data.copy()


# In[16]:


data.shape


# In[17]:


data.dtypes


# In[18]:


data.head()


# In[19]:


data.corr()


# In[20]:


y = data['Loan_Status']
x = data.drop(['Loan_ID', 'Loan_Status'], axis=1)


# In[21]:


x.head()


# In[22]:


y.head()


# In[23]:


x.shape


# In[24]:


x = pd.get_dummies(x)


# In[25]:


x.shape


# In[26]:


x.head()


# ## DATA MODELLING (RANDOM FOREST CLASSIFIER)

# In[27]:


from scipy.stats import entropy
entropy([5, 7])


# In[27]:


#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=45)


# In[28]:


#x_train.shape


# In[29]:


#x_test.shape


# In[30]:


#y_train.shape


# In[31]:


#y_test.shape


# ## HYPERPARAMETER TUNING - RANDOMIZED SEARCH

# In[28]:


#FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train = sc.fit_transform(x)


# In[29]:


from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 55)
rfmodel.fit(x, y)


# In[30]:


from sklearn.model_selection import RandomizedSearchCV

rfmodel1 = RandomForestClassifier(n_jobs = -1)
n_estimators = [10, 100, 200, 300, 400, 500, 700, 900, 1000]
criterion = ['gini', 'entropy']
max_depth = [3, 5, 10]
parameters = {'n_estimators':n_estimators, 'criterion':criterion, 'max_depth':max_depth}


# In[31]:


def hypertuning_rscv(rfmodel1, p_distr, nbr_iter, x, y):
    RFC_rfmodel1 = RandomizedSearchCV(rfmodel1, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=10)
    RFC_rfmodel1.fit(x, y)
    ht_params = RFC_rfmodel1.best_params_
    ht_score = RFC_rfmodel1.best_score_
    return ht_params, ht_score


# In[32]:


rf_parameters, rf_ht_score = hypertuning_rscv(rfmodel1, parameters, 40, x, y)


# In[33]:


rf_parameters


# In[34]:


rf_ht_score


# In[35]:


rfmodel = RandomForestClassifier(n_jobs = -1, n_estimators = 700, criterion = 'gini', max_depth = 3)


# In[36]:


rfmodel.fit(x, y)


# In[37]:


#ypred_rf1 = rfmodel.predict(x)


# ## SMOTE - RANDOM FOREST

# In[38]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 55)
x_res, y_res = sm.fit_resample(x, y)


# In[39]:


#FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_res = sc.fit_transform(x_res)


# In[40]:


from sklearn.ensemble import RandomForestClassifier

rfmodel2 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 55)
rfmodel2.fit(x_res, y_res)


# In[41]:


from sklearn.model_selection import RandomizedSearchCV

rfmodel3 = RandomForestClassifier(n_jobs = -1)
n_estimators = [10, 100, 200, 300, 400, 500, 700, 900, 1000]
criterion = ['gini', 'entropy']
max_depth = [3, 5, 10]
parameters = {'n_estimators':n_estimators, 'criterion':criterion, 'max_depth':max_depth}


# In[42]:


def hypertuning_rscv(rfmodel3, p_distr, nbr_iter, x_res, y_res):
    RFC_rfmodel3 = RandomizedSearchCV(rfmodel3, param_distributions=p_distr, n_jobs=-1, n_iter=nbr_iter, cv=10)
    RFC_rfmodel3.fit(x_res, y_res)
    ht_params = RFC_rfmodel3.best_params_
    ht_score = RFC_rfmodel3.best_score_
    return ht_params, ht_score


# In[43]:


rf_parameters, rf_ht_score = hypertuning_rscv(rfmodel3, parameters, 40, x_res, y_res)


# In[61]:


rf_parameters


# In[62]:


rf_ht_score


# In[64]:


rfmodel2 = RandomForestClassifier(n_jobs = -1, n_estimators = 500, criterion = 'entropy', max_depth = 10)


# In[65]:


rfmodel2.fit(x, y)


# ## TEST DATA

# In[66]:


test = pd.read_csv("/home/adarsh/Documents/MACHINE LEARNING - PROJECTS/Data/Loan_Prediction/test.csv")


# In[67]:


test.head()


# In[68]:


test.isna().sum()


# In[69]:


test.shape


# In[70]:


test.dtypes


# In[71]:


test['Gender'][test['Gender'].isna()] = test['Gender'].mode()[0]


# In[72]:


test['Dependents'][test['Dependents'].isna()] = test['Dependents'].mode()[0]


# In[73]:


test['Self_Employed'][test['Self_Employed'].isna()] = test['Self_Employed'].mode()[0]


# In[74]:


test['LoanAmount'][test['LoanAmount'].isna()] = test['LoanAmount'].mean()


# In[75]:


test['Loan_Amount_Term'][test['Loan_Amount_Term'].isna()] = test['Loan_Amount_Term'].mean()


# In[76]:


test['Credit_History'][test['Credit_History'].isna()] = test['Credit_History'].mean()


# In[77]:


test.isna().sum()


# In[78]:


test.head()


# In[79]:


test_x = test.drop('Loan_ID', axis = 1)


# In[80]:


test_x


# In[81]:


test_x = pd.get_dummies(test_x)


# In[82]:


y_prediction = rfmodel2.predict(test_x)


# In[83]:


y_prediction


# In[84]:


my_submission = pd.DataFrame({'Loan_ID': test.Loan_ID, 'Loan_Status': y_prediction})


# In[85]:


my_submission.to_csv('submission_RF3.csv', index = False)

