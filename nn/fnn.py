import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Module to display 'ex3data1.csv'
from displayex3 import displayData, displayImage

def sigmoid(z):
	return 1/(1+np.exp(-z))

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

# Add the bias array to 'X'
X = np.column_stack((np.ones(m), X))

# Define the number of classes
classes = 10

### Verify the dimensions and type of 'X', 'y', and 'theta'
### Note: X includes the bias vector already
print("X:", X.shape, type(X))
print("y:", y.shape, type(y))

### Load the weight vectors of Neural Network
Theta1 = np.loadtxt("ex3weights1.csv", delimiter=",")
Theta2 = np.loadtxt("ex3weights2.csv", delimiter=",")
### Verify the dimensions and type of 'Theta1' and 'Theta2'
print("Theta1:", Theta1.shape, type(Theta1))
print("Theta2:", Theta2.shape, type(Theta2))

### Determine the second activation function
a2 = sigmoid(np.dot(X,Theta1.T))
### Verify 'a2'
print("a2:", a2.shape, type(a2))
### Add the bias vector to 'a2'
a2 = np.insert(a2, 0, np.ones((1,m)), axis=1)
### Re-verify 'a2' and check that the bias vector is in the first column
print("a2:", a2.shape, type(a2))
print("a2[0]:", a2[0])
### Determine the their activiation function (the hypothesis)
a3 = sigmoid(np.dot(a2,Theta2.T))
### Verify 'a3'
print("a3:", a3.shape, type(a3))
print("a3[0]:", a3[0])

### 'a3' now contain the 5000x10 hypothesis matrix
### Each row corresponds to an image
### Each column corresponds to the hypothesis for each image
### Checking the accuracy
correct = 0
for i, hyp, img in zip(range(len(y)), a3, y):
	if (int(img)%10 == int(str(np.argmax(hyp)+1)[-1])):
		# print("Success!")
		correct += 1
	else:
		print("#{} (guess: {}, actual: {})".format(i, int(str(np.argmax(hyp))[-1]), int(img)%10))

### Display the final results!
print("{} right out of 5000 ({}%)".format(correct, '%0.1f'%(correct/5000*100)))