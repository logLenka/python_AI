import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("linreg_data.csv",skiprows=0,names=["x","y"])
# print(data.head())

xpd = data["x"]
ypd = data["y"]
n = xpd.size
# plt.scatter(xpd,ypd)
# plt.show()

xbar = np.mean(xpd)
ybar = np.mean(ypd)

term1 = np.sum(xpd*ypd)
term2 = np.sum(xpd**2)

b = (term1-n*xbar*ybar)/(term2-n*xbar*xbar)
a = ybar - b*xbar

x = np.linspace(0,2,100)
y = a+b*x
plt.plot(x,y,color="black")
# plt.plot(xpd,y,color="black") - ValueError
plt.scatter(xpd,ypd)
plt.scatter(xbar,ybar,color="red")
plt.show()

yhat = a+b*xpd

xval = np.array([0.5,0.75,0.90])
# xval = [0.5,0.75,0.90] - TypeError when calculating yval
yval = a+b*xval
print(xval)


df = pd.read_csv("weight-height.csv")
print(df.corr())
