# -*- coding: utf-8 -*-
"""Py4DS_Lab2_ex3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wHgrPnoHVK6SiyRKtCKh04Qg84SOZJQi
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Datasets/Py4DS_Lab2/creditcard.csv")
data = data.drop(['Time'],axis=1)
V = data.drop(['Amount','Class'],axis = 1)
data

for i in range(0,data.shape[1]):
  print(data.iloc[:,i].value_counts())
  print("*"*30)

for i in range(0,data.shape[1]):
  print(data.iloc[:,i].describe())
  print('*'*30)

plt.figure(figsize=(20,20))
Vcorr = V.corr().round(2)
sns.heatmap(Vcorr,cmap='YlGnBu',annot=True)
plt.show()
print('Du lieu cac cot khonog co su lien quan den nhau')

P_static = sns.countplot(data = data, x = "Class")
plt.show()
print('Du lieu khong co su can doi, 0 qua nhieu, 1 qua it')

Amount = sns.violinplot(x="Class",y="Amount",data=data)
plt.show()
print('Du lieu 0 qua lon va co outlayer kha lon')
print('Du lieu 1 qua nho va tap trung 1 cho')

Amount = sns.boxplot(x="Class",y="Amount",data=data)
plt.show()
print('boxplot nhin xau hon violin plot nhung ta van co the ket luan tuong tu tren')

Amount = sns.boxplot(data.Amount)
plt.show()
print('bieu do chi cho amout thoi, Du lieu co outlayer rat lon, gay nhieu')

