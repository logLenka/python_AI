

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn import metrics

# data = np.genfromtxt("weight-height.csv", delimiter=",")
data = pd.read_csv("weight-height.csv",skiprows=1, names=["Gender","Height","Weight"])
# height = data["Height"]
length = np.array(data["Height"]).reshape(-1,1)
weight = np.array(data["Weight"]).reshape(-1,1)

regr = linear_model.LinearRegression()
regr.fit(length, weight)

xval = length
yval = regr.predict(xval)
plt.plot(xval, yval)
plt.scatter(length, weight, color="red")
plt.show()

yhat = regr.predict(xval)
yp = weight
print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))  
print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))
print('R2 value:', metrics.r2_score(yp, yhat))


