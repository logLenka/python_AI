"""
scores for lienar regression:
    rmse = 3.229701118499755; r2 = 0.8129145926619961
    rmse_test = 4.130335091127992; r2_test = 0.7705165707309942
lasso --> optimal alpha = 0.3 (with the highest score: 0.7750)
    LASSO regressor returns zero values for cylinders, displacement, horsepower, acceleration
ridge --> optimal alpha= 93.87755102040816  (with the highest r2 = 0.7717097331686409)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# import seaborn as sns


df = pd.read_csv('Auto.csv')
# print(df.head())
# sns.heatmap(data=df.corr().round(2), annot=True)
# plt.show()

X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year']]
y = df[['mpg']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# linear regression
lm = LinearRegression()
lm.fit(X_train, y_train)

y_train_predict = lm.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

y_test_predict = lm.predict(X_test)
rmse_test = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2_test = r2_score(y_test, y_test_predict)

print(rmse,r2)
print(rmse_test,r2_test)


# LASSO regression
alphas = [0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8]
scores = []
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)
    lasso.fit(X_train, y_train)
    # print(lasso.coef_.round(2))
    sc = lasso.score(X_test, y_test)
    scores.append(sc)
    # print("alpha=",alp," lasso score:", sc)

# print(max(scores))
plt.plot(alphas, scores)
plt.xlabel("alpha")
plt.ylabel("R2 score")
plt.show()

# ridge regression
alphas = np.linspace(0,200,50) # past 4 the score decreases
r2values = []
for alp in alphas:
    rr = Ridge(alpha=alp)
    rr.fit(X_train, y_train)
    r2_test = r2_score(y_test, rr.predict(X_test))
    r2values.append(r2_test)
    # print("alpha=",alp," r2:", r2_test)

# print(max(r2values))
plt.plot(alphas,r2values)
plt.show()