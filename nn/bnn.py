import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Module to display 'ex3data1.csv'
from displayex3 import displayData, displayImage

def sigmoid(z):
	return 1/(1+np.exp(-z))

def costFunction(Theta, ninput, nhidden, nclasses, X, y, lamb=0, check=False):
	hyp = lambda z: sigmoid(z)

	[m,n] = X.shape
	### Add the bias array to 'X'
	X = np.column_stack((np.ones(m), X))
	
	### Unflatten 'Theta' into 'Theta1' and 'Theta2'
	Theta1 = Theta[:(nhidden*(ninput+1))].reshape(nhidden, (ninput+1))
	Theta2 = Theta[(nhidden*(ninput+1)):].reshape(nclasses, (nhidden+1))

	### Check the dimensionality
	if (check):
		print("X:", X.shape, type(X))
		print("y:", y.shape, type(y))
		print("Theta1:", Theta1.shape, type(Theta1))
		print("Theta2:", Theta2.shape, type(Theta2))
	###############

	# cost = np.zeros((5000, nclasses))
	cost = 0

	### Determine the second activation function
	a2 = sigmoid(np.dot(X,Theta1.T))
	print("a2:", a2.shape, type(a2))
	### Add the bias vector to 'a2'
	a2 = np.insert(a2, 0, np.ones((1,m)), axis=1)
	### Re-verify 'a2' and check that the bias vector is in the first column
	print("a2:", a2.shape, type(a2))

	### Determine the their activiation function (the hypothesis)
	a3 = sigmoid(np.dot(a2,Theta2.T))
	print("a3:", a3.shape, type(a3))

	### working for-loop
	cost = np.zeros(m)
	for i in range(m):
		yi = int(y[[i]])
		cat = [1 if (val == yi) else 0 for val in range(1,nclasses+1)]
		cat = np.array((cat)).reshape(1,nclasses)
		# print(int(yi), cat)
		cost[i] += np.sum(np.multiply(-cat,np.log(a3[[i]])) - np.multiply((1-cat),np.log(1-a3[[i]])), axis=1)
		# print("cost[{}] = {}".format(i, cost[i]))
	print(np.average(cost, axis=0))
	# print(np.average(cost, axis=0))
	# print(cost)

	# return np.average(np.multiply(-y,np.log(hyp(X))) - np.multiply((1-y),np.log(1-hyp(X))), axis=0)

### "ex3data1.csv" contains 5,000 samples of 401 element arrays
### These correspond to 20x20 pixel images (400 elements) and 1 classification
### The images are written numbers and the classifications are the number (0-9)
X = np.loadtxt(os.path.join("..", "data", "ex3data1.csv"), delimiter=",", skiprows=1)
# Assign 'y' as the classification array
y = X[:,[-1]]
# Remove 'y' from X
X = np.delete(X, -1, axis=1)
[m,n] = X.shape

### Display the data
# displayData(X)
# displayImage(X, 1500)
##############

# Define the number of classes (k)
k = 10
# ### Assume weight matricies of random floats between -epsilon and +epsilon
# epsilon = 0.1
# s1, s2, s3 = n, 25, k
# Theta1 = np.subtract(np.multiply(2*epsilon, np.random.rand(s2,s1+1)), epsilon)
# [m1,n1] = Theta1.shape
# Theta2 = np.subtract(np.multiply(2*epsilon, np.random.rand(s3,s2+1)), epsilon)
# [m2,n2] = Theta2.shape

# # Remap 'y' from ('1', '2', ..., '9', '10') to '1's and '0's
# y = [1 if (int(val)%10 == i) else 0 for val in y]

### Load the weight vectors of Neural Network to verify the cost function
Theta1 = np.loadtxt("ex3weights1.csv", delimiter=",")
Theta2 = np.loadtxt("ex3weights2.csv", delimiter=",")
##### Cost should be 0.287629

### Flatten into a single theta matrix
Theta = np.concatenate((Theta1.ravel(), Theta2.ravel()))

## ### Determine the second activation function
## a2 = sigmoid(np.dot(X,Theta1.T))
## ### Verify 'a2'
## print("a2:", a2.shape, type(a2))
## ### Add the bias vector to 'a2'
## a2 = np.insert(a2, 0, np.ones((1,m)), axis=1)
## ### Re-verify 'a2' and check that the bias vector is in the first column
## print("a2:", a2.shape, type(a2))
## print("a2[0]:", a2[0])
## ### Determine the their activiation function (the hypothesis)
## a3 = sigmoid(np.dot(a2,Theta2.T))
## ### Verify 'a3'
## print("a3:", a3.shape, type(a3))
## print("a3[0]:", a3[0])

### Verify the dimensions and type of 'X', 'y', and the weight matricies
print("X:", X.shape, type(X))
print("y:", y.shape, type(y))
print("Theta1:", Theta1.shape, type(Theta1))
print("Theta2:", Theta2.shape, type(Theta2))
print("Theta:", Theta.shape, type(Theta))

costFunction(Theta, 400, 25, 10, X, y, check=True)
print("Should be 0.287629")

##### Test case
### https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
# il = 2
# hl = 2
# nl = 4
# nn = (np.arange(1,18+1)/10).reshape(18,1)
# X = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
# y = np.array([[4], [2], [3]])

# print("X:", X.shape, type(X))
# print("y:", y.shape, type(y))
# print("Theta:", nn.shape, type(nn))

# costFunction(nn, il, hl, nl, X, y, check=True)
# print("Should be 7.4070")
########################

# # Loop over the first
# cost = 0
# for i in range(k):
# 	cat = [1 if (int(val)%10 == i) else 0 for val in y]
# 	cat = np.array((cat)).reshape(m,1)
# 	cost += costFunction(Theta1, X, cat, check=True)
# 	a2 = sigmoid(np.dot(X,Theta1.T))
# 	a2 = np.insert(a2, 0, np.ones((1,m)), axis=1)
# 	cost += costFunction(Theta2, a2, cat, check=True)

# print(cost)

