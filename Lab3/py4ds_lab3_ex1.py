# -*- coding: utf-8 -*-
"""Py4DS_Lab3_ex1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UFy_2xqiJhy3wJq19u3pDK0iiEhVNiez
"""


import numpy as np
import pandas as pd

df = pd.read_csv('creditcard.csv')
df

X = df.drop(['Time','Class'], axis = 1)
Y = df['Class']

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.9)

"""Scaling X with StandardScaler"""

scaler = preprocessing.StandardScaler()
X_train_ss = scaler.fit_transform(X_train)
X_test_ss = scaler.transform(X_test)

"""Scaling X with MinMaxScaler"""

scaler = preprocessing.MinMaxScaler()
X_train_mms = scaler.fit_transform(X_train)
X_test_mms = scaler.transform(X_test)

"""Scaling X with MaxAbsScaler"""

scaler = preprocessing.MaxAbsScaler()
X_train_mas = scaler.fit_transform(X_train)
X_test_mas = scaler.transform(X_test)

"""Scaling X with RobustScaler"""

scaler = preprocessing.RobustScaler()
X_train_rs = scaler.fit_transform(X_train)
X_test_rs = scaler.transform(X_test)

"""Scaling X with PCA"""

from sklearn.decomposition import PCA
scaler = PCA(n_components = "mle", svd_solver = "full")
X_train_pca = scaler.fit_transform(X_train)
X_test_pca = scaler.transform(X_test)

"""Mapping X with Uniform distribution"""

transformer = preprocessing.QuantileTransformer()
X_train_u = transformer.fit_transform(X_train)
X_test_u = transformer.transform(X_test)

"""Mapping X with Gaussion distribution(Yeo-Johnson)"""

transformer = preprocessing.PowerTransformer(method = "yeo-johnson", standardize=False)
X_train_g = transformer.fit_transform(X_train)
X_test_g = transformer.transform(X_test)

"""Mapping X with Normalize distribution"""

transformer = preprocessing.Normalizer(norm = 'l2')
X_train_n = transformer.transform(X_train)
X_test_n = transformer.transform(X_test)

"""Encoder X with KBin Discretization"""

encoder = preprocessing.KBinsDiscretizer(n_bins = 10, strategy = 'uniform')
X_train_kbin = encoder.fit_transform(X_train)
X_test_kbin = encoder.transform(X_test)

"""Function to classify"""

def LRaccuracy(X_train,X_test,Y_train,Y_test):
  import time
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score
  start = time.time()
  clf = LogisticRegression(max_iter = X_train.shape[0]).fit(X_train,Y_train)
  Y_predict = clf.predict(X_test)
  score = accuracy_score(Y_test,Y_predict)
  end = time.time()
  print("Accuracy score for Logistic Regression: ",score)
  print("Time estimated: ",end-start)
  return score

def SVCaccuracy(X_train,X_test,Y_train,Y_test):
  import time
  from sklearn import svm
  from sklearn.metrics import accuracy_score
  start = time.time()
  clf = svm.SVC()
  clf.fit(X_train,Y_train)
  Y_predict = clf.predict(X_test)
  score = accuracy_score(Y_test,Y_predict)
  end = time.time()
  print("Accuracy score for Support Vector Classification: ",score)
  print("Time estimated: ",end-start)
  return score

def DTCaccuracy(X_train, X_test,Y_train,Y_test):
  import time
  from sklearn import tree
  from sklearn.metrics import accuracy_score
  start = time.time()
  clf = tree.DecisionTreeClassifier()
  clf.fit(X_train,Y_train)
  Y_predict = clf.predict(X_test)
  score = accuracy_score(Y_test,Y_predict)
  end = time.time()
  print("Accuracy score for Decision Tree Classification: ", score)
  print("Time estimatimated: ",end-start)
  return score

def RFCaccuracy(X_train,X_test,Y_train,Y_test):
  import time
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.metrics import accuracy_score
  start = time.time()
  clf = RandomForestClassifier()
  clf.fit(X_train,Y_train)
  Y_predict = clf.predict(X_test)
  score = accuracy_score(Y_test,Y_predict)
  end = time.time()
  print("Accuracy score for Random Forest Classifier: ", score)
  print("Time estimated: ",end-start)
  return score

def Classify(X_train,X_test,Y_train,Y_test):
  lr = LRaccuracy(X_train,X_test,Y_train,Y_test)
  svc = SVCaccuracy(X_train,X_test,Y_train,Y_test)
  dtc = DTCaccuracy(X_train,X_test,Y_train,Y_test)
  rfc = RFCaccuracy(X_train,X_test,Y_train,Y_test)
  print("*"*50)
  return np.array([lr,svc,dtc,rfc])

def Compare(NoneProcessing, Processing):
  for i in range(len(NoneProcessing)):
    state = ""
    if (NoneProcessing[i]<Processing[i]):
      state = "Good"
    else:
      state = "Bad"
    switcher = {
        0: "Logistic Regression",
        1: "Support Vector Classification",
        2: "Decision Tree Classification",
        3: "Random Forest Classification"
    }
    print(state + " " + switcher.get(i))
  print("*"*50)
  print("*"*50)

"""Compare before and after preprocessing"""

print("Classify without preprocessing: ")
scores_before = Classify(X_train,X_test,Y_train,Y_test)

print("Classify with StandardScaler: ")
scores_after = Classify(X_train_ss,X_test_ss,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with MinMaxScaler: ")
scores_after = Classify(X_train_mms,X_test_mms,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with MaxAbsScaler: ")
scores_after = Classify(X_train_mas,X_test_mas,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with RobustScaler: ")
scores_after = Classify(X_train_rs,X_test_rs,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with PCA")
scores_after = Classify(X_train_pca,X_test_pca,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with Uniform distribution")
scores_after = Classify(X_train_u,X_test_u,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with Gauss distribution")
scores_after = Classify(X_train_g,X_test_g,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with Nomalize distribution")
scores_after = Classify(X_train_n,X_test_n,Y_train,Y_test)
Compare(scores_before,scores_after)

print("Classify with KBinDiscrete")
scores_after = Classify(X_train_kbin,X_test_kbin,Y_train,Y_test)
Compare(scores_before,scores_after)

