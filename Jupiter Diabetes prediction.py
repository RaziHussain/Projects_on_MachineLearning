#!/usr/bin/env python
# coding: utf-8

# In[54]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[55]:


data = pd.read_csv("D:\pimadata.csv")


# In[56]:


data.shape


# In[57]:


data.head()


# In[58]:


# checking if any null values are present or not
data.isnull().values.any()


# In[59]:


#Correlation
import seaborn as sns
import matplotlib.pyplot as plt
# get Correlation of each features in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (20,20))
#plot heat map
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap="Spectral")


# In[60]:


data.corr()


# In[61]:


top_corr_features = corrmat.index
print(top_corr_features)


# In[62]:


#Changing the diabetes column data from boolean to number

diabetes_map = {True: 1, False:0}


# In[63]:


data['diabetes'] = data['diabetes'].map(diabetes_map)


# In[64]:


data.head()


# In[65]:


diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])


# In[66]:


(diabetes_true_count, diabetes_false_count)


# In[67]:


#Train Test Split

from sklearn.model_selection import train_test_split
feature_columns = ['num_preg' , 'glucose_conc' , 'diastolic_bp', 'bmi', 'diab_pred', 'age', 'skin' ]
predicted_class = ['diabetes']


# In[68]:


X = data[feature_columns].values
Y = data[predicted_class].values

X_train, X_test , Y_train, Y_test = train_test_split(X , Y , test_size = 0.30, random_state= 10)


# In[69]:


#Cheching how many other missing(zero) values

print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


# In[70]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)


# In[71]:


print(X_train)


# In[72]:


print(X_test)


# In[73]:


## Apply Algorithm

from sklearn.ensemble import RandomForestClassifier
random_forest_model = RandomForestClassifier(random_state=10)

random_forest_model.fit(X_train, Y_train.ravel())


# In[74]:


predict_train_data = random_forest_model.predict(X_test)

from sklearn import metrics

print("Accuracy = {0:.3f}".format(metrics.accuracy_score(Y_test, predict_train_data)))


# In[75]:


from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[76]:


#getting the Statistical measures of the data
data.describe()


# In[77]:


data['diabetes'].value_counts()


# In[78]:


data.groupby('diabetes').mean()


# In[79]:


#Separating Data and Labels
A = data.drop(columns = 'diabetes' , axis =1)


# In[80]:


B = data['diabetes']


# In[81]:


print(A)
print(B)


# In[82]:


#Data Standardization
scaler = StandardScaler()


# In[83]:


scaler.fit(A)


# In[84]:


standardized_data = scaler.transform(A)


# In[85]:


print(standardized_data)


# In[86]:


A = standardized_data
B = data['diabetes']


# In[87]:


print(A)
print(B)


# In[88]:


#Train Test Split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.3, stratify = B, random_state = 2)


# In[89]:


print(A.shape, A_train.shape, A_test.shape)


# In[90]:


#Training the Model
classifier = svm.SVC(kernel='linear')


# In[126]:


poly_classifier = svm.SVC(kernel ='poly', degree = 20)
poly_classifier.fit(A_train, B_train) 
# training set in x, y axis


# In[133]:


sig_classifier = svm.SVC(kernel ='sigmoid')


# In[134]:


#training the svm classifier
classifier.fit(A_train, B_train)


# In[128]:


poly_classifier.fit(A_train, B_train)


# In[135]:


sig_classifier.fit(A_train, B_train)


# In[136]:


#Model Evaluation --> Accuracy Score
#accuracy on the training data
A_train_prediction = classifier.predict(A_train)
training_data_accuracy = accuracy_score(A_train_prediction, B_train)


# In[137]:


A_train_prediction_poly = poly_classifier.predict(A_train)
training_data_accuracy_poly = accuracy_score(A_train_prediction_poly, B_train)


# In[138]:


A_train_prediction_sig = sig_classifier.predict(A_train)
training_data_accuracy_sig = accuracy_score(A_train_prediction_sig, B_train)


# In[139]:


print('Accuracy Score on the training data: ', training_data_accuracy)


# In[140]:


print('Accuracy Score on the training data: ', training_data_accuracy_poly)


# In[141]:


print('Accuracy Score on the training data: ', training_data_accuracy_sig)


# In[41]:


#accuracy on the test data
A_test_prediction = classifier.predict(A_test)
test_data_accuracy = accuracy_score(A_test_prediction, B_test)


# In[42]:


print('Accuracy Score on the test data: ', test_data_accuracy)


# In[43]:


#Making a Predictive System

input_data = (5,166,72,19,175,25.8,0.587,51,0.7486)

#Changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#We need to reshape the array as we are predicting for one instance
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

#Hence we already Did the Standardization of data earlier So we have to Standardize this input_data
std_data = scaler.transform(input_data_reshape)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The Person is non Diabetic')
else:
    print('The Person is Diabetic')


# In[44]:


#Hyper Parameter Optimization

params={
    "learning_rate" :[0.05,0.10,0.15,0.20,0.25,0.30],
    "max_depth":[3,4,5,6,8,10,12,15],
    "min_child_weight":[1,3,5,7],
    "gamma":[0.0,0.1,0.2,0.3,0.4],
    "colsample_bytree":[0.3,0.4,0.5,0.7]
    }


# In[45]:


#Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
import xgboost


# In[46]:


classifier=xgboost.XGBClassifier( missing=1, seed=42) 


# In[47]:


random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=150,scoring='roc_auc', n_jobs=-1,cv=5,verbose=2)


# In[48]:


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour,temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time Taken: %i hours %i minutes and %s seconds.' %(thour,tmin,round(tsec, 2)))


# In[49]:


from datetime import datetime
#here we go
start_time=timer(None)  #timing starts from this point for "start_time" variable
random_search.fit(X_train,Y_train.ravel())
timer(start_time) #timing ends here for "start_time" variable


# In[50]:


random_search.best_estimator_


# In[51]:


'''classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=0.5,
              enable_categorical=False, gamma=0.4, gpu_id=-1,
              importance_type=None, interaction_constraints='',
              learning_rate=0.05, max_delta_step=0, max_depth=15,
              min_child_weight=3, missing=None, monotone_constraints='()',
              n_estimators=100, n_jobs=4, num_parallel_tree=1, predictor='auto',
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method='exact', validate_parameters=1,
              verbosity=None)'''


# In[52]:


classifier.fit(X_train, Y_train.ravel())


# In[53]:


y_pred=classifier.predict(X_test)


# In[54]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test, y_pred)
score= accuracy_score(Y_test, y_pred)

print(cm)
print(score)


# In[55]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(classifier, X_train, Y_train.ravel(),cv=10)


# In[56]:


score


# In[57]:


score.mean()


# In[ ]:


#K Nearest Neighbor Algorithm
#Pros:
#No assumptions about data
#Simple algorithm — easy to understand
#Can be used for classification and regression

#Cons:
#High memory requirement — All of the training data must be present in memory in order to calculate the closest K neighbors
#Sensitive to irrelevant features
#Sensitive to the scale of the data since we’re computing the distance to the closest K points


# In[58]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()


# In[273]:


#Train Test Split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = 0.25, random_state = 0)


# In[ ]:





# In[274]:


knn = KNeighborsClassifier(n_neighbors=7)


# In[275]:


knn.fit(A_train, B_train)


# In[276]:


# Predict on dataset which model has not seen before
print(knn.predict(A_test))


# In[277]:


# Calculate the accuracy of the model
print(knn.score(A_test, B_test))


# In[290]:


neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[291]:


#aneighbors = np.arange(1,9)
#print(neighbors)


# In[292]:


# Loop over K values
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(A_train, B_train)


# In[293]:


# Compute training and test data accuracy
train_accuracy[i] = knn.score(A_train, B_train)
test_accuracy[i] = knn.score(A_test, B_test)


# In[294]:


# Generate plot
plt.plot(neighbors, test_accuracy, label = 'Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training dataset Accuracy')
 
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[295]:


from sklearn.linear_model import LogisticRegression
cl = LogisticRegression()
cl.fit(A_train, B_train)


# In[296]:


B_pred = cl.predict(A_test)


# In[297]:


print ("Accuracy : ", accuracy_score(B_test, B_pred))


# In[298]:


from sklearn.tree import DecisionTreeClassifier


# In[299]:


dtree = DecisionTreeClassifier()
dtree = dtree.fit(A_train, B_train)


# In[300]:


d1=dtree.predict(A_test)


# In[301]:


print("Accuracy : ", accuracy_score(B_test, d1))


# In[ ]:





# In[ ]:




