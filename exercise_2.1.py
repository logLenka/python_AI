# Draw the lines y = 2x + 1, y = 2x + 2 and y = 2x + 3 in the same figure. 
# Use different drawing colors and line types for your graphs to make them 
# stand out in black and white. Set the image title and captions 
# for the horizontal and vertical axes.

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 7, 100)
y1 = 2*x + 1
y2 = 2*x + 2
y3 = 2*x + 3
plt.plot(x,y1,color="red")
plt.plot(x,y2,color="green", linestyle="--")
plt.plot(x,y3,color="blue", linestyle="dashdot")
plt.title("3 parallel lines")
plt.xlabel("x")
plt.ylabel("y")
plt.show()