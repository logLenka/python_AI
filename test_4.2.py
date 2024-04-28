import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import metrics

my_data = np.genfromtxt('linreg_data.csv', delimiter=',')
xp = my_data[:,0]
yp = my_data[:,1]
xp = xp.reshape(-1,1)
yp = yp.reshape(-1,1)


regr = linear_model.LinearRegression()
regr.fit(xp, yp) # fitting the model=training the model

print(regr.coef_,regr.intercept_)

xval = np.full((1,1),0.5)
yval = regr.predict(xval)

# xval = np.linspace(-1,2,20).reshape(-1,1)
xval = xp
yval = regr.predict(xval)
plt.plot(xval,yval) # this plots the line
plt.scatter(xp,yp,color="red")
plt.show()

yhat = regr.predict(xp)
print('Mean Absolute Error:', metrics.mean_absolute_error(yp, yhat))  
print('Mean Squared Error:', metrics.mean_squared_error(yp, yhat))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yp, yhat)))
print('R2 value:', metrics.r2_score(yp, yhat))