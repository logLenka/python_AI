# Calculate the inverse matrix of matrix A and  check with the matrix product 
# that both AA−1 and A−1A produce a unit matrix with ones in diagonals and zeros elsewhere.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

A = np.array([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
Ainv = np.linalg.inv(A)

# product1 = np.dot(A, Ainv)
# product2 = np.dot(Ainv, A)

product1 = np.matmul(A, Ainv)
product2 = np.matmul(Ainv, A)

print("product1:", product1)
print("product2:", product2)







