import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Module to display 'ex3data1.csv'
from displayex3 import displayData, displayImage

### "ex3data1.csv" contains 5,000 samples of 401 element arrays
### These correspond to 20x20 pixel images (400 elements) and 1 classification
### The images are written numbers and the classifications are the number (0-9)
X = np.loadtxt(os.path.join("..", "data", "ex3data1.csv"), delimiter=",", skiprows=1)
[m,n] = X.shape
# Assign 'y' as the classification array
y = X[:,[-1]]
# Remove 'y' from X
X = np.delete(X, -1, axis=1)

### Display the data
# displayData(X)
# displayImage(X, 1500)
##############