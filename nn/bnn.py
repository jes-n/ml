import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Module to display 'ex3data1.csv'
from displayex3 import displayData, displayImage

def sigmoid(z):
	return 1/(1+np.exp(-z))

def sigprime(z):
	return sigmoid(z)*(1-sigmoid(z))

def costFunction(Theta, ninput, nhidden, nclasses, X, y, lamb=0, check=False):
	### Adjust the feature matrix to ensure that the dimensions are (# features)x(# data points)
	[n,m] = X.shape
	if (n != ninput):
		print("Warning: Reshaping 'X' to be (# features)x(# data points)")
		X = X.T
		[n,m] = X.shape
	elif (n == m):
		print("Warning: 'X' is a square matrix, ensure it is of dimensions (# features)x(# data points)")
	### Add the bias array to 'X'
	X = np.concatenate((np.ones((1,m)), X))
	
	### Unflatten 'Theta' into 'Theta1' and 'Theta2'
	Theta1 = Theta[:(nhidden*(ninput+1))].reshape((ninput+1), nhidden)
	Theta2 = Theta[(nhidden*(ninput+1)):].reshape((nhidden+1), nclasses)

	if (check):
		print("X:", X.shape, type(X))
		print("y:", y.shape, type(y))
		print("Theta:", Theta.shape, type(Theta))
		print("Theta1:", Theta1.shape, type(Theta1))
		print("Theta2:", Theta2.shape, type(Theta2))

	### Determine the second activation function
	a2 = sigmoid(np.dot(Theta1.T,X))
	a2 = np.concatenate((np.ones((1,m)), a2)) # Add the bias vector to 'a2'
	### Determine the third activiation function (the hypothesis)
	a3 = sigmoid(np.dot(Theta2.T,a2))
	
	### Check the dimensionality
	if (check):
		print("a2:", a2.shape, type(a2))
		print("a3:", a3.shape, type(a3))
	###############

	### Vectorize the classification for each data point
	cat = np.zeros((nclasses,m))
	for i in range(m):
		yi = int(y[i])
		cat[:,i] = [1 if (val == yi) else 0 for val in range(1,nclasses+1)]
	y = cat
	if (check):
		print("y:", y.shape, type(y))

	### Forward Propagate to determine the cost
	cost = np.sum(np.multiply(-y,np.log(a3)) - np.multiply((1-y),np.log(1-a3)), axis=0)
	if (check):
		print("Cost:", cost.shape, type(cost))

	### Back Propagate to determine the deltas
	# print(d3.shape, d3[[i]].shape)
	# d3[[i]] = a3[[i]] - cat
	# print(Theta2.T.shape, d3[[i]].T.shape, a2.T.shape, Theta2.shape)
	# d2[[i]] = np.multiply(np.multiply(Theta2.T, d3[[i]].T), sigprime(np.dot(a2[[i]],Theta2.T)))

	### Apply Regularization
	regularization = (lamb/(2*m))*(np.sum(np.power(Theta1[1:,:],2)) + np.sum(np.power(Theta2[1:,:],2)))

	return np.average(cost) + regularization

### "ex3data1.csv" contains 5,000 samples of 401 element arrays
### These correspond to 20x20 pixel images (400 elements) and 1 classification
### The images are written numbers and the classifications are the number (0-9)
X = np.loadtxt(os.path.join("..", "data", "ex3data1.csv"), delimiter=",", skiprows=1)
### Transform 'X' to follow standard notation of (# features)x(# data points)
X = X.T
### Assign 'y' as the classification vector
y = X[-1]
### Remove 'y' from X
## In NumPy: 'axis=0' corresponds to the rows and 'axis=1' corresponds to the columns
X = np.delete(X, -1, axis=0)
## 'n' is the # of features and 'm' is the # of data points
[n,m] = X.shape
## 'k' is the # of unique classes
k = len(np.unique(y))

### Verify the loading and parsing of data
print("{} features for {} data points with {} unique classes\n".format(n,m,k))
### Verify the dimensions and type of 'X' and 'y'
# print("X:", X.shape, type(X))
# print("y:", y.shape, type(y))

### Display the data
# displayData(X)
# displayImage(X, 1500)
##############

### Assign the # of units in each layer
## 's1' is the # of features
## 's2' is the # of units in the hidden layer
## 's3' is the # of classes
s1, s2, s3 = n, 25, k
### Assume weight matricies of random floats between -(epsilon) and +(epsilon)
epsilon = 0.12
Theta1 = np.subtract(np.multiply(2*epsilon, np.random.rand(s2,s1+1)), epsilon)
Theta2 = np.subtract(np.multiply(2*epsilon, np.random.rand(s3,s2+1)), epsilon)
### Flatten into a single weight vector to pass to the cost function
Theta = np.concatenate((Theta1.ravel(), Theta2.ravel()))

##### Test case using weight vectors from 'ex3'
# Theta1 = np.loadtxt("ex3weights1.csv", delimiter=",").T
# Theta2 = np.loadtxt("ex3weights2.csv", delimiter=",").T
# Theta = np.concatenate((Theta1.ravel(), Theta2.ravel()))
# print("Unregularized 'ex2weights' test")
# cost = costFunction(Theta, s1, s2, s3, X, y, check=True)
# print("Cost:", cost, "(should be 0.287629)\n")
# print("Regularized 'ex2weights' test")
# cost = costFunction(Theta, s1, s2, s3, X, y, lamb=1, check=True)
# print("Cost:", cost, "(should be 0.383770)\n")
##############

##### Test case for 'ex4'
##### https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
# il = 2
# hl = 2
# nl = 4
# nn = np.arange(1,18+1)/10
# X = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
# y = np.array([[4], [2], [3]])
# print("Unregularized 'ex4' test (from forum)")
# cost = costFunction(nn, il, hl, nl, X, y, check=True)
# print("Cost:", cost, "(should be 7.4070)\n")
# print("Regularized 'ex4' test (from forum)")
# cost = costFunction(nn, il, hl, nl, X, y, lamb=4, check=True)
# print("Cost:", cost, "(should be 19.474)\n")
##############

# ### Verify the dimensions and type of 'X', 'y', and the weight matricies
# print("X:", X.shape, type(X))
# print("y:", y.shape, type(y))
# print("Theta1:", Theta1.shape, type(Theta1))
# print("Theta2:", Theta2.shape, type(Theta2))
# print("Theta:", Theta.shape, type(Theta))

### Determine the weight vector for each classification
###################
### This will be done by iterating over each class
### A Conjugate Gradient minimization will be preformed to determine each set of weight vectors
###################
start = time.time() # For minimization timing purposes
cost = costFunction(Theta, s1, s2, s3, X, y, check=True)
print("Time elapsed:", time.time() - start, "seconds") # Show the elapsed time
