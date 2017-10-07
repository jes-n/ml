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
	[m,n] = X.shape
	### Add the bias array to 'X'
	X = np.column_stack((np.ones(m), X))
	
	### Unflatten 'Theta' into 'Theta1' and 'Theta2'
	Theta1 = Theta[:(nhidden*(ninput+1))].reshape(nhidden, (ninput+1))
	Theta2 = Theta[(nhidden*(ninput+1)):].reshape(nclasses, (nhidden+1))

	### Determine the second activation function
	a2 = sigmoid(np.dot(X,Theta1.T))
	a2 = np.column_stack((np.ones(m), a2)) # Add the bias vector to 'a2'
	print(a2)
	### Determine the third activiation function (the hypothesis)
	a3 = sigmoid(np.dot(a2,Theta2.T))
	print(a3)
	
	### Check the dimensionality
	if (check):
		print("X:", X.shape, type(X))
		print("y:", y.shape, type(y))
		print("Theta:", Theta.shape, type(Theta))
		print("Theta1:", Theta1.shape, type(Theta1))
		print("Theta2:", Theta2.shape, type(Theta2))
		print("a2:", a2.shape, type(a2))
		print("a3:", a3.shape, type(a3))
	###############

	### Sum the cost function over all classes
	cost = np.zeros(m)
	d2 = np.zeros(m)
	d3 = np.zeros(m)
	for i in range(m):
		yi = int(y[[i]])
		cat = [1 if (val == yi) else 0 for val in range(1,nclasses+1)]
		cat = np.column_stack(np.array((cat)))
		# print(int(yi), cat)
		# print(cat.shape, a3[[i]].shape)
		# print(cat, 1-cat, a3[[i]])
		# print(i)
		cost[i] += np.sum(np.multiply(-cat,np.log(a3[[i]])) - np.multiply((1-cat),np.log(1-a3[[i]])), axis=1)
		# print("cost[{}] = {}".format(i, cost[i]))
		d3 = a3[[i]] - cat
		d2 = np.multiply(np.multiply(Theta2.T, d3[[i]]), sigprime(np.dot(a2,Theta2.T)))

	### Apply Regularization
	reg = (lamb/2/m)*(np.sum(np.sum(np.power(Theta1[:,1:],2), axis=1), axis=0) + np.sum(np.sum(np.power(Theta2[:,1:],2), axis=1), axis=0))

	return np.average(cost, axis=0) + reg

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
### Assume weight matricies of random floats between -epsilon and +epsilon
epsilon = 0.12
s1, s2, s3 = n, 25, k
Theta1 = np.subtract(np.multiply(2*epsilon, np.random.rand(s2,s1+1)), epsilon)
[m1,n1] = Theta1.shape
Theta2 = np.subtract(np.multiply(2*epsilon, np.random.rand(s3,s2+1)), epsilon)
[m2,n2] = Theta2.shape
## Flatten into a single theta matrix
Theta = np.concatenate((Theta1.ravel(), Theta2.ravel()))

##### Test case using weight vectors from 'ex3'
# Theta1 = np.loadtxt("ex3weights1.csv", delimiter=",")
# Theta2 = np.loadtxt("ex3weights2.csv", delimiter=",")
# Theta = np.concatenate((Theta1.ravel(), Theta2.ravel()))
# cost = costFunction(Theta, s1, s2, s3, X, y, check=True)
# print("Cost:", cost, "(should be 0.287629)")
# cost = costFunction(Theta, s1, s2, s3, X, y, lamb=1, check=True)
# print("Cost:", cost, "(should be 0.383770)")
##############

##### Test case for 'ex4'
##### https://www.coursera.org/learn/machine-learning/discussions/weeks/5/threads/uPd5FJqnEeWWpRIGHRsuuw
# il = 2
# hl = 2
# nl = 4
# nn = np.arange(1,18+1)/10
# X = np.cos(np.array([[1, 2], [3, 4], [5, 6]]))
# y = np.array([[4], [2], [3]])
# cost = costFunction(nn, il, hl, nl, X, y, check=True)
# print("Cost:", cost, "(should be 7.4070)")
# cost = costFunction(nn, il, hl, nl, X, y, lamb=4, check=True)
# print("Cost:", cost, "(should be 19.474)")
##############

### Verify the dimensions and type of 'X', 'y', and the weight matricies
print("X:", X.shape, type(X))
print("y:", y.shape, type(y))
print("Theta1:", Theta1.shape, type(Theta1))
print("Theta2:", Theta2.shape, type(Theta2))
print("Theta:", Theta.shape, type(Theta))

### Determine the weight vector for each classification
###################
### This will be done by iterating over each class
### A Conjugate Gradient minimization will be preformed to determine each set of weight vectors
###################
# start = time.time() # For minimization timing purposes
# for i in range(classes):
# 	print("Learning {}".format(i))
# 	yi = int(y[[i]])
# 	cat = [1 if (val == yi) else 0 for val in range(1,nclasses+1)]
# 	cat = np.column_stack(np.array((cat)))
# 	cat = [1 if (int(val)%10 == i) else 0 for val in y]
# 	cat = np.array((cat)).reshape(m,1)
# 	res = minimize(costFunction, theta[[i]], args=(X, cat),
# 		method='BFGS', options={"disp":True, "gtol":5e-4}, callback=callback)
# 	print(res.message)
# 	# print("Weight vector for '{}':".format(i), res.x)
# 	print("Minimum cost for '{}':".format(i), costFunction(res.x.reshape(1,n), X, cat))
# 	theta[i,:] = res.x.reshape(1,n)
# print("Time elapsed:", time.time() - start, "seconds") # Show the elapsed time
# ### Save the weight vector 'theta' for later use because it is computationally expensive
# np.savetxt("weights.csv", theta, delimiter=",")

