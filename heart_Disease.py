#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


data=pd.read_csv("C:/Users/drkir/Downloads/heart.csv")
data.head()


# In[3]:


len(data)


# In[4]:


# num of col and row 
data.shape


# In[5]:


# some information about dataset
data.info()


# In[6]:


#statistic of dataset
data.describe()


# In[7]:


# check duplication 
data.duplicated().sum()


# In[8]:


# drop duplication 
data.drop_duplicates(inplace=True)


# In[9]:


# check after duplication 
data.duplicated().sum()


# In[10]:


data.isnull().sum()


# In[11]:


data.nunique()


# In[12]:


data.columns


# In[13]:


for col in data.columns:
    print (col, ":" , data[col].nunique())
    print (data[col].value_counts().nlargest(5))


# In[14]:


sns.pairplot(data=data)
plt.show()


# In[15]:


data.hist(figsize=(12,12),layout=(5,3))
plt.show()


# In[16]:


data["sex"].value_counts()


# In[17]:


#How many values there in two categoris --> we can use value_counts() function  ,, result-> male>female 
sns.countplot(x='sex',data=data)
plt.show()


# In[18]:


sns.countplot(x='sex',data=data,hue='target')
plt.show()


# In[19]:


#Here 1 means male and 0 denotes female. we observe female having heart disease are comparatively less when compared to males Males have low heart diseases as compared to females in the given dataset.


# In[20]:


sns.catplot(data=data, x='sex', y='age',  hue='target', palette='husl')
plt.show()


# In[21]:


sns.barplot(data=data, x='sex', y='chol', hue='target', palette='husl')
plt.show()


# In[22]:


# There are 160 people suffering from heart disease and those who do not suffer from heart disease are 140
sns.countplot(x='target',palette='spring', data=data)
plt.show()


# In[23]:


sns.countplot(x='ca',hue='target',data=data)
plt.show()


# In[24]:


sns.countplot(x='thal',data=data, hue='target', palette='BuPu' )
plt.show()


# In[25]:


sns.displot(data['target'],kde =True,height=3,aspect=3)
plt.xlabel("target")
plt.ylabel("frequancy")
plt.title("Distribution of target")
plt.show()


# In[26]:


data.target.skew()


# In[27]:


# check outliers
plt.figure(figsize=(7,3))
sns.boxplot(x='chol', data=data)
plt.show()


# In[28]:


# drop outlier points from 'chol'.
data=data[data['chol'] <= 370]


# In[29]:


plt.figure(figsize=(7,3))
sns.boxplot(x='chol', data=data)
plt.show()


# In[30]:


plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot=True, cmap='terrain')
plt.show()


# In[31]:


sns.barplot(x='fbs', y='chol', hue='target', data=data,palette='plasma' )
plt.show()


# In[32]:


sns.barplot(x='sex',y='target', hue='fbs',data=data)
plt.show()


# In[33]:


gen = pd.crosstab(data['sex'], data['target'])
print(gen)


# In[34]:


gen.plot(kind='bar', stacked=True, grid=False)
plt.show()


# In[35]:


#SPLITTING AND SCALING DATA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
StandardScaler = StandardScaler()  
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
data[columns_to_scale] = StandardScaler.fit_transform(data[columns_to_scale])


# In[36]:


data.head()


# In[37]:


X= data.drop(['target'], axis=1)
y= data['target']


# In[38]:


X_train, X_test,y_train, y_test=train_test_split(X,y,test_size=0.3,random_state=40)


# In[39]:


data.head()


# In[40]:


print('X_train-', X_train.size)
print('X_test-',X_test.size)
print('y_train-', y_train.size)
print('y_test-', y_test.size)


# In[41]:


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

model1=lr.fit(X_train,y_train)
prediction1=model1.predict(X_test)


# In[42]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,prediction1)
cm


# In[43]:


sns.heatmap(cm, annot=True,cmap='BuPu')
plt.show()


# In[44]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction1)


# In[45]:


#DECISION TREE
from sklearn.tree import DecisionTreeClassifier

dtc=DecisionTreeClassifier()
model2=dtc.fit(X_train,y_train)
prediction2=model2.predict(X_test)
cm2= confusion_matrix(y_test,prediction2)
cm2


# In[46]:


accuracy_score(y_test,prediction2)


# In[47]:


#RANDOMFOREST 
from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
model3 = rfc.fit(X_train, y_train)
prediction3 = model3.predict(X_test)
confusion_matrix(y_test, prediction3)


# In[48]:


accuracy_score(y_test, prediction3)


# In[49]:


#SVM
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


# In[50]:


svm=SVC()
model4=svm.fit(X_train,y_train)
prediction4=model4.predict(X_test)
cm4= confusion_matrix(y_test,prediction4)
cm4


# In[51]:


accuracy_score(y_test, prediction4)


# In[52]:


#NAIVE BAYES
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
model5 = NB.fit(X_train, y_train)
prediction5 = model5.predict(X_test)
cm5= confusion_matrix(y_test, prediction5)
cm5


# In[53]:


accuracy_score(y_test, prediction5)


# In[54]:


#KNN
from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()
model6 = KNN.fit(X_train, y_train)
prediction6 = model6.predict(X_test)
cm6= confusion_matrix(y_test, prediction5)
cm6


# In[55]:


print('KNN :', accuracy_score(y_test, prediction6))
print('lr :', accuracy_score(y_test, prediction1))
print('dtc :', accuracy_score(y_test, prediction2))
print('rfc :', accuracy_score(y_test, prediction3))
print('NB: ', accuracy_score(y_test, prediction4))
print('SVC :', accuracy_score(y_test, prediction5))


# In[ ]:




