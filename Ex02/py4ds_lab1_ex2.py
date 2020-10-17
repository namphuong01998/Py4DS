# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np

path = './diabetes.csv'
df = pd.read_csv(path)
df

# Kiem tra du lieu khuyet
print(df.info())

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age']
X = df[feature_cols]
y = df['Outcome']

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)

"Decision Tree Classification"

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
print(eli5.format_as_text(eli5.explain_weights(perm)))

"Support Vector Classification"


print("Support Vector Classification")
from sklearn.svm import SVC
clf = SVC()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print('Final Score: ',scores.mean())

"Random Forest Classification"

print('Random Forest Classification')
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print('Accuracy: ',accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print("Final Score: ",scores.mean())

"Logistic Regression"

print("Logistic Regression")
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter = np.size(y_train))
clf = clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)
print("Acurracy: ",accuracy_score(y_test,y_pred))

scores = cross_val_score(clf,X,y,cv=5)
print("Scores: ",scores)
print("Final score: ",scores.mean())

