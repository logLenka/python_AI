'''a) next I would add PTRATIO as it has the 3rd strongest correlation with the target variable (MEDV)

b) scores with added PTRATIO:
        RMSE = 5.30386319330793; R2= 0.6725214335656512
        RMSE (test) = 4.913937534764078; R2 (test)= 0.6915878280744177
    vs. scores without PTRATIO
        RMSE= 5.6371293350711955; R2= 0.6300745149331701
        RMSE (test)= 5.137400784702911; R2 (test)= 0.6628996975186952

c) adding more variables does not help, there is not a strong correlation between MEDV & the rest of variables'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.datasets import fetch_openml
# from sklearn.datasets import fetch_california_housing

# data = fetch_california_housing()  
# data = fetch_openml(name="house_prices", as_frame=True)
# data = load_boston()
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :3]])
target = raw_df.values[1::2, 2]
# print(data.keys())
# print(data.DESCR)


# df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.DataFrame(data, columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "MEDV"])
# df['MEDV'] = data.target
print(df.head())
plt.hist(df['MEDV'],25)
plt.xlabel("MEDV")
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1,3,1)
plt.scatter(df['RM'],df['MEDV'])
plt.xlabel("RM")
plt.ylabel("MEDV")

plt.subplot(1,3,2)
plt.scatter(df['LSTAT'],df['MEDV'])
plt.xlabel("LSTAT")
plt.ylabel("MEDV")

plt.subplot(1,3,3)
plt.scatter(df['PTRATIO'],df['MEDV'])
plt.xlabel("PTRATIO")
plt.ylabel("MEDV")
plt.show()

X = pd.DataFrame(df[['RM','LSTAT', 'PTRATIO']], columns = ['RM','LSTAT', 'PTRATIO'])
y = df['MEDV']
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