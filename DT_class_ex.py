#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# Deepak Agarwal------>Email:deepakagarwal@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 June - 05 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Decision Tree  %%%%%%%%%%%%%%%%%%%%%%%%%%

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#%%-----------------------------------------------------------------------

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser



# Exercise
#%%-----------------------------------------------------------------------
# Specify what are your features and targets. Why this is a classification
# 1- Use the bank banknote dataset.
# 2- Specify what are your features and targets.
# 3- Why this is a classification problem.
# 4- Run the decision tree algorithm.
# 5- Explain your findings and write down a paragraph to explain all the results.
#%%-----------------------------------------------------------------------
# 1-
banknote = pd.read_csv("data_banknote_authentication.data", header=-1)

banknote.columns = ['C1', 'C2', 'C3', 'C4', 'Target']

banknote['Target'] = banknote['Target'].astype(str)

print(banknote.head(10))


#%%-----------------------------------------------------------------------
# 2-

# We will specify the features as the first 4 columns and the target as the 5th column

#%%-----------------------------------------------------------------------
# 3-

# Column 5 is comprised of classes (0s and 1s)

#%%-----------------------------------------------------------------------
# 4-
X = banknote.values[:, 0:4]
y = banknote.values[:, 4]

#-----------------------------------------------------------------------------
# data pre processing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#%%-----------------------------------------------------------------------
# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)
#%%-----------------------------------------------------------------------
# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)
#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = banknote.Target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
#%%-----------------------------------------------------------------------
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = banknote.Target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=banknote.iloc[:, 0:4].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini1.pdf")
webbrowser.open_new(r'decision_tree_gini1.pdf')

#%%-----------------------------------------------------------------------
# display decision tree

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names=banknote.iloc[:, 0:4].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy1.pdf")
webbrowser.open_new(r'decision_tree_entropy1.pdf')

print ('-'*40 + 'End Console' + '-'*40 + '\n')


#%%-----------------------------------------------------------------------
# 5-
#The accuracy for Gini Index is 94.174 and the accuracy for enropy is 94.66. 
#There are two target variables, 0 and 1 and four features. 
#For entropy the number of false negatives is 0, 
#but the number of false positives is higher than that of gini index. 
#For the target value is 1, the prediction value in gini index is higher 
#than that of entropy. While for the target value 0, 
#the precision value of entropy is higher than that of gini index. 
#The recall for target value 0 is almost the same, while for target value 1, 
#recall is higher in entropy. 
#The formula for F1 score is: 
#   F1 = 2 * (precision * recall) / (precision + recall), 
#thus the values for both target values are not significantly different. 



# In[2]:


# %%%%%%%%%%%%% Machine Learning %%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Authors  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Dr. Amir Jafari------>Email: amir.h.jafari@okstate.edu
# Deepak Agarwal------>Email:deepakagarwal@gwmail.gwu.edu
# %%%%%%%%%%%%% Date:
# V1 June - 05 - 2018
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%% Decision Tree  %%%%%%%%%%%%%%%%%%%%%%%%%%
#%%-----------------------------------------------------------------------
#%%-----------------------------------------------------------------------
# Exercise
#%%-----------------------------------------------------------------------

# 1:
# Build the simple tennis table we just reviewed, in python as a dataframe. Label the columns.
# We are going to calculate entropy manually, but in python.
# Make sure to enter all variables as binary vs. the actual categorical names
# Name the dataframe tennis_ex.
#%%-----------------------------------------------------------------------
from sklearn.preprocessing import LabelEncoder 
import pandas as pd 
import numpy as np
tennis_ex=pd.read_csv("tennis.csv")

tennis_ex.columns=['Outlook', 'Temperature', 'Humidity','Windy','Target']
cleanup_nums = {"Outlook":     {"sunny": 0, "overcast": 1,"rainy":2},
                "Temperature": {"hot": 0, "mild": 1, "cool": 2},
                "Humidity" :   {"high": 0, "normal": 1 }}
tennis_ex['Target'] = tennis_ex['Target'].astype(str)
tennis_ex.replace(cleanup_nums, inplace=True)

  
le = LabelEncoder() 
  
tennis_ex['Outlook']= le.fit_transform(tennis_ex['Outlook']) 
tennis_ex['Temperature']= le.fit_transform(tennis_ex['Temperature'])
tennis_ex['Humidity']= le.fit_transform(tennis_ex['Humidity']) 
tennis_ex['Windy']= le.fit_transform(tennis_ex['Windy']) 
 
#%%-----------------------------------------------------------------------
# 2:
# Build a function that will calculate entropy. Calculate entropy for the table we just went over
# in the example, but in python
# This is for the first split.
#%%-----------------------------------------------------------------------
from math import log, e

def entropy3(labels, base=None):
    vc = pd.Series(labels).value_counts(normalize=True, sort=False)
    base = e if base is None else base
    return -(vc * np.log2(vc)/np.log2(base)).sum()

for i in range(0,4,1):
    labels = tennis_ex.iloc[:,i]
    print(entropy3(labels))
    
    
# 3:
# Run the decision tree algorithm and find out the best feature and graph it.
#%%-----------------------------------------------------------------------
# Importing the required packages
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import sys

#%%-----------------------------------------------------------------------
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
#%%-----------------------------------------------------------------------

# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser
#%%--------------------------------Save Console----------------------------

# old_stdout = sys.stdout
# log_file = open("console.txt", "w")
# sys.stdout = log_file

#%%-----------------------------------------------------------------------


X = tennis_ex.values[:, 0:4]
y = tennis_ex.values[:, 4]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", splitter="best" ,random_state=100, max_depth=5, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)


#%%-----------------------------------------------------------------------
# perform training with entropy.
# Decision tree with entropy
clf_entropy = DecisionTreeClassifier(criterion="entropy",splitter="best" ,random_state=100, max_depth=5, min_samples_leaf=5)

# Performing training
clf_entropy.fit(X_train, y_train)

# make predictions
# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

# predicton on test using entropy
y_pred_entropy = clf_entropy.predict(X_test)

#%%-----------------------------------------------------------------------
# calculate metrics gini model
print("\n")
print("Results Using Gini Index: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_gini))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
print("\n")
print ('-'*80 + '\n')
# calculate metrics entropy model
print("\n")
print("Results Using Entropy: \n")
print("Classification Report: ")
print(classification_report(y_test,y_pred_entropy))
print("\n")
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
print ('-'*80 + '\n')
#%%-----------------------------------------------------------------------
conf_matrix = confusion_matrix(y_test, y_pred_gini)
class_names = tennis_ex.Target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )
#%%-----------------------------------------------------------------------
plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------

# confusion matrix for entropy model
conf_matrix = confusion_matrix(y_test, y_pred_entropy)
class_names = tennis_ex.Target.unique()
df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))
hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)
hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.tight_layout()
plt.show()

#%%-----------------------------------------------------------------------
# display decision tree
dot_data = export_graphviz(clf_gini, filled=True, rounded=True, class_names=class_names, feature_names=tennis_ex.iloc[:, 0:4].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_gini2.pdf")
webbrowser.open_new(r'decision_tree_gini2.pdf')

#%%-----------------------------------------------------------------------
# display decision tree

dot_data = export_graphviz(clf_entropy, filled=True, rounded=True, class_names=class_names, feature_names=tennis_ex.iloc[:, 0:4].columns, out_file=None)

graph = graph_from_dot_data(dot_data)
graph.write_pdf("decision_tree_entropy2.pdf")
webbrowser.open_new(r'decision_tree_entropy2.pdf')

print ('-'*40 + 'End Console' + '-'*40 + '\n')


# In[ ]:




# In[ ]:




