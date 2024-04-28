"""scores:
        R2= 0.8196352304254052
        R2 (norm)= 0.8233786447294953
        R2 (std)= 0.8236711987425501
the best R2 value is for standardized data"""

import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
import numpy as np

df = pd.read_csv('weight-height.csv',skiprows=0,delimiter=",")
# print(df)

X = df[["Height"]]
y = df[["Weight"]]
# print(X)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=30)

X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)
# print(X_train_norm)
# print(X_test_norm)
# print(X_train_std)
# print(X_test_std)

lm = neighbors.KNeighborsRegressor(n_neighbors=5)
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print("R2=",lm.score(X_test,y_test))

lm.fit(X_train_norm, y_train)
#predictions2 = lm.predict(X_test_norm)
print("R2 (norm)=",lm.score(X_test_norm,y_test))

lm.fit(X_train_std, y_train)
#predictions3 = lm.predict(X_test_std)
print("R2 (std)=",lm.score(X_test_std,y_test))

