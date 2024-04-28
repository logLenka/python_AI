

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

n = 100000
s = np.array([])
for i in range(n):
    a, b = np.random.randint(1, 7, 2)
    sum = a+b
    s = np.append(s, sum)
h,h2 = np.histogram(s,range(2,14))
# print(h)
plt.bar(h2[:-1],h/n)
plt.show()

# print(a, b)







