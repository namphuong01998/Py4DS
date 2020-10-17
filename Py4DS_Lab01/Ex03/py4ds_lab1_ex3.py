# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

path = './mushrooms.csv'
df = pd.read_csv(path)
df

# Kiem tra du lieu khuyet
print(df.info())

import sklearn
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
for columns in df.columns:
  df[columns] = labelEncoder.fit_transform(df[columns])
df=df.drop(['veil-type'],axis=1)

X = df.drop('class',axis=1)
y = df['class']
print(X)
print(y)

df.describe()

df_div = pd.melt(df,'class',var_name='Characteristics')
df_div

import matplotlib
#import matplotlib.pyplot as plt
import pylab as plt
import seaborn as sns
#import matplotlib.axes.Axes.set_xticklabels
fig, ax = plt.subplots(figsize=(10,5))
chart = sns.violinplot(x = 'Characteristics',y = 'value',hue = 'class',data = df_div,split=True)
df_no_class = df.drop(["class"],axis = 1)
chart.set_xticklabels(rotation = 90, labels=list(df_no_class))

#import matplotlib.pyplot as plt
plt.figure()
#pd.Series(df['class']).value_counts()
pd.Series(df['class']).value_counts().sort_index().plot(kind = 'bar')
plt.ylabel("Count")
plt.xlabel("class")

plt.figure(figsize=(14,12))
sns.heatmap(df.corr(),linewidths=1,cmap="YlGnBu",annot=True)
plt.yticks(rotation = 0)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

"""Decision Tree Classification"""

print("Decided Tree Classification")
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print('Scores: ',scores)
print('Final Score: ',scores.mean())

import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(clf,random_state=1).fit(X_test,y_test)
eli5.show_weights(perm,feature_names = X_test.columns.to_list())


"""Support Vector Classification"""

print("Support Vector Classification")
from sklearn.svm import SVC
clf = SVC()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print('Final Score: ',scores.mean())

"""Random Forest Classification"""

print('Random Forest Classification')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print("Final Score: ",scores.mean())

"""Logistic Regression"""

print("Logistic Regression")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter = np.size(y_train))
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Acurracy: ",accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print("Final score: ",scores.mean())

