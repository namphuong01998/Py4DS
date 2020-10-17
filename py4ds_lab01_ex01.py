# -*- coding: utf-8 -*-


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"""# **STEP 1: Read and view file**"""

spams = pd.read_csv(r"./spam_original.csv")
spams.head()

"""# **STEP 2: Kiểm tra dữ liệu lỗi**"""

spams.info()

"""# **STEP 3**: 
##**1. Tách X va y (dataframe)**
"""

X = spams.iloc[:, :-1]               ## X là dữ liệu bỏ đi cột cuối cùng Outcome và lấy tất cả các dòng trong table
y = np.array(spams.iloc[:, -1]);     ## y thanh array

print(y[:5])
X.head()

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

"""**Comments:**

Từ confusion matrix chỉ ra rằng trong tổng số 1381 dữ liệu của test-set thì có:

*   ####  Trong y_test: 
   - spam = 0 : 
      - Thực tế: 837
      - Dự đoán chính xác:  828
      - Dự đoán sai: 9
    - spam = 1 : 
      - Thực tế: 544
      - Dự đoán chính xác:  458
      - Dự đoán sai: 86

* #### Bài toán: accuracy  = 66,184%; 
==> Chủ yếu dự đoán sai nhóm spam = 1.

##**2. Model: Logistic Regression**
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = LogisticRegression(max_iter = len(y_train))
clf.fit(X_train,y_train)

print("Logistic Regression")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))

"""**Comments:**

Từ confusion matrix chỉ ra rằng trong tổng số 1381 dữ liệu của test-set thì có:

*   ####  Trong y_test: 
   - spam = 0 : 
      - Thực tế: 837
      - Dự đoán chính xác: 792
      - Dự đoán sai: 45
    - spam = 1 : 
      - Thực tế: 544
      - Dự đoán chính xác:  53
      - Dự đoán sai: 491

* #### Bài toán: accuracy  = 92,904%; 
==> Chủ yếu dự đoán sai nhóm spam = 1.

#**3. Model: Decision Tree Classifier**
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

print("Decision Tree Classifier")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))

"""**Comments:**

Từ confusion matrix chỉ ra rằng trong tổng số 1381 dữ liệu của test-set thì có:

*   ####  Trong y_test: 
   - spam = 0 : 
      - Thực tế: 837
      - Dự đoán chính xác: 781
      - Dự đoán sai: 56
    - spam = 1 : 
      - Thực tế: 544
      - Dự đoán chính xác:  51
      - Dự đoán sai: 493

* #### Bài toán: accuracy  = 92,252%; 
==> Chủ yếu dự đoán sai nhóm spam = 1.

#**4. Model: Random Forest Classifier**
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

clf = RandomForestClassifier()
clf.fit(X_train,y_train)

print("Random Forest Classifier")
print("\ntrain acc: ", accuracy_score(y_train, clf.predict(X_train)))
print("train acc: ", accuracy_score(y_test, clf.predict(X_test)))
print('\nConfusion matrix : \n', confusion_matrix(y_test, clf.predict(X_test)))
print('Classification report : \n', classification_report(y_test, clf.predict(X_test), digits = 5))

"""**Comments:**

Từ confusion matrix chỉ ra rằng trong tổng số 1381 dữ liệu của test-set thì có:

*   ####  Trong y_test: 
   - spam = 0 : 
      - Thực tế: 837
      - Dự đoán chính xác: 811
      - Dự đoán sai: 26
    - spam = 1 : 
      - Thực tế: 544
      - Dự đoán chính xác:  32
      - Dự đoán sai: 512

* #### Bài toán: accuracy  = 95,8%; 
==> Chủ yếu dự đoán sai nhóm spam = 1.
"""
