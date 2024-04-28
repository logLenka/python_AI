# #Read from CSV-file weight-height.csv to numpy-table information about the lengths and weights 
# (in inches and pounds) of a group of students. Collect the lengths for the variable "length" and the weights for the variable "weight" by cutting the table.
# Convert lengths from inches to centimeters and weights from pounds to kilograms.
# Finally, calculate the means, medians, standard deviations, and variances of the lengths and weights.
# Extra: Draw a histogram pattern of the lengths

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data = np.genfromtxt("weight-height.csv", delimiter=",")
data = pd.read_csv("weight-height.csv",skiprows=1, names=["Gender","Height","Weight"])
# height = data["Height"]
length = np.array(data["Height"])
weight = np.array(data["Weight"])
length_cm = length * 2.54
weight_kg = weight * 0.45
length_cm_Mean = np.mean(length_cm)
weight_kg_Mean = np.mean(weight_kg)
length_cm_Median = np.median(length_cm)
weight_kg_Median = np.median(weight_kg)
length_cm_Deviation = np.std(length_cm)
weight_kg_Deviation = np.std(weight_kg)
length_cm_Variation = np.var(length_cm)
weight_kg_Variation = np.var(weight_kg)
print(length_cm_Deviation)
print(length_cm_Variation)

plt.hist(length_cm)
plt.title("lengths in cm")
plt.show()
