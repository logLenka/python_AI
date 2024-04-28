# Look for a command from the numpy documentation (or elsewhere on the Internet) to calculate 
# the matrix determinant. The determinant is a number that does not require a deeper 
# understanding here. Calculate the determinants of the matrices A, B and AB, where
# A =
# B =
# and AB stands for matrix product. State that det (AB) = det (A) det (B) can be 
# understood as the product of the numbers (the rounding error can be ignored).

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

a = np.array([[1, 2], [3, 4]])
b = np.array([[-1, 1], [5, 7]])
# AB = a * b
AB = np.dot(a,b)


det_A = np.linalg.det(a)
det_B = np.linalg.det(b)
det_AB = np.linalg.det(AB)
det_AB2 = det_A * det_B
# print(AB)
# print(det_A)
print("det_AB:", det_AB)
print("det_AB2:", det_AB2)



