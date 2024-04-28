"""3) appropriate variables to predict company profit: 'R&D Spend','Marketing Spend' as they have the strongest correlation with the target variable (Profit).
However there is also some multicollinearity present - high values of correlation between explanatory variables.
The resul shows high R2 values (0.9436, 0.9684) for training as well as for testing data. """

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("50_Startups.csv", sep=",")
data = pd.read_csv("50_Startups.csv", sep=",")

print(data.keys())



print(df.head())
# plt.hist(df['MEDV'],25)
# plt.xlabel("MEDV")
# plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,2,1)
plt.scatter(df['R&D Spend'],df['Profit'])
plt.xlabel("R&D Spend")
plt.ylabel("Profit")

plt.subplot(1,2,2)
plt.scatter(df['Marketing Spend'],df['Profit'])
plt.xlabel("Marketing Spend")
plt.ylabel("Profit")

plt.show()

X = pd.DataFrame(df[['R&D Spend','Marketing Spend']], columns = ['R&D Spend','Marketing Spend'])
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)

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