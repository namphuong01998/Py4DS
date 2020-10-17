# -*- coding: utf-8 -*-

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"""# **STEP 1: Read and view file**"""

API_edu = pd.read_csv(r"./xAPI-Edu-Data.csv")
API_edu.head()

"""# **STEP 2: Kiểm tra dữ liệu lỗi**"""

API_edu.info()

"""# **STEP 3**: 
##**1. Tách X va y (dataframe)**
"""

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Tạo biến target
y = API_edu['Class']

#  Tập X được chia thành 2 loại:
## Data có dữ liệu là continuous (Dữ liệu chứa số)
X_continuous = API_edu[['Discussion', 'AnnouncementsView', 'VisITedResources', 'raisedhands']]
## Data với dữ liệu là category (Dữ liệu chứa chữ)
X_category = API_edu.drop(['Class', 'Discussion', 'AnnouncementsView', 'VisITedResources', 'raisedhands'], axis = 1)
## Encoding các data
X_category_encode = pd.get_dummies(X_category, prefix_sep='_')
y = le.fit_transform(y)
X = pd.concat([X_continuous, X_category_encode], axis = 1)

"""##**2. Train_Test**"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    stratify = y, random_state = 0)

"""#**STEP 4: Choose model**

##**1. Model: Support Vector Machine (SVM)**
"""

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf_svm = SVC(kernel = 'poly', C = 10)   
clf_svm.fit(X_train, y_train)

print("Support Vector Machine")
print("\ntrain acc: ", accuracy_score(y_train, clf_svm.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf_svm.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf_svm.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf_svm.predict(X_test), digits = 5))

"""#**2. Model: Decision Tree Classifier**"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print("Decision Tree Classifier")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))

"""#**3. Model: Random Forest Classifier**"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print("Random Forest Classifier")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))
