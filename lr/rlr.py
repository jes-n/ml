import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def sigmoid(z):
	return 1/(1+np.exp(-z))

def _costFunctionReg(theta, X, y, lamb=0):
	_hyp = lambda X: sigmoid(np.dot(X,theta.T))
	theta = theta.reshape(1,X.shape[1])
	return np.average(np.multiply(-y,np.log(_hyp(X))) - np.multiply((1-y),np.log(1-_hyp(X)))) + (lamb/(2*X.shape[1]))*np.sum(np.power(np.delete(theta, 0, axis=1),2))

def costFunctionReg(Theta, X, y, lamb=0, check=False):
	### Adjust the feature matrix to ensure that the dimensions are (# features)x(# data points)
	[n,m] = X.shape
	# if (n != ninput):
	# 	print("Warning: Reshaping 'X' to be (# features)x(# data points)")
	# 	X = X.T
	# 	[n,m] = X.shape
	# elif (n == m):
	# 	print("Warning: 'X' is a square matrix, ensure it is of dimensions (# features)x(# data points)")
	### Add the bias array to 'X'
	X = np.concatenate((np.ones((n,1)), X), axis=1)
	if (check):
		print("X:", X.shape, type(X))
		print("y:", y.shape, type(y))
		print("Theta", Theta.shape, type(Theta))

	hyp = lambda Theta, X: sigmoid(np.dot(Theta.T,X))
	cost = (np.multiply(-y,np.log(hyp(Theta,X))) - np.multiply((1-y),np.log(1-hyp(Theta,X))))
	print(X[:,0].shape, Theta.T.shape, X.shape)
	print(hyp(Theta,X).shape, sigmoid(np.dot(Theta.T,X)).shape)
	print((hyp(Theta,X)-y).shape)
	grad = []
	for j in range(m+1):
		grad.append(np.average(np.multiply(hyp(Theta,X)-y, X[:,j]), axis=1))

	# print(np.power(Theta[1:],2).shape)
	regularization = (lamb/(2*m))*(np.sum(np.power(Theta[1:],2)))

	return np.average(cost) + regularization, 0

	# theta = theta.reshape(1,X.shape[1])
	# cost = _costFunctionReg(theta, X, y)
	# grad = []
	# for i in range(X.shape[1]):
	# 	grad.append(np.average(np.multiply((hyp(X)-y), X[:,[i]])))
	# return cost, grad

def mapFeature(X, degree=6):
	""" Maps a feature vector onto a given number of degrees. 
	The new number of features will be (n+1)(n+2)/2,
	where n is the degree.
	This is known as the 'nth triangular number'.
	This can be quickly written in Python as the sum(range(n+2))
	"""
	dim = 1
	[m,n] = X.shape
	mapped = np.ones((m,sum(range(degree+2))))
	for i in range(1, degree+1):
		for j in range(i+1):
			mapped[:,dim] = np.multiply(np.power(X[:,0], (i-j)), np.power(X[:,1], j))
			dim +=1
	return mapped

X = np.loadtxt(os.path.join("..", "data", "ex2data2.csv"), delimiter=",")
[m,n] = X.shape
y = X[:,[-1]]
X = np.delete(X, -1, axis=1)

theta = np.zeros((m,1))

print("X:", X.shape, type(X))
print("y:", y.shape, type(y))
print("Theta", theta.shape, type(theta))

# X = mapFeature(X, degree=6)
# print("X:", X.shape, type(X))

cost, grad = costFunctionReg(theta, X, y, lamb=0, check=True)
print("Cost:", cost)
print("Expected cost (approx): 0.693")
print("Grad:", grad)
print("Expected gradients (approx): [-0.1000, -12.0092, -11.2628]")

##### Unregularized test case
# X = np.array(([8, 1, 6],
# 			  [3, 5, 7],
# 			  [4, 9, 2]))
# y = np.array((1, 0, 1))
# Theta = np.array((-2, -1, 1, 2))
# print("Unregularized test case (lamb=0)")
# cost, grad = costFunctionReg(Theta, X.T, y, lamb=0, check=True)
# print("\tCost:", cost)
# print("\tExpected cost (approx): 4.6832")
# ##### Regularized test case
# print("Regularized test case (lamb=4)")
# cost, grad = costFunctionReg(Theta, X.T, y, lamb=4, check=True)
# print("\tCost:", cost)
# print("\tExpected cost (approx): 8.6832")
##########

# res = minimize(_costFunctionReg, theta, args=(X, y),
# 	method='BFGS', options={"maxiter":500, "disp":True}, callback=None)

# print("\nSolution:", res.x, "with cost =", _costFunctionReg(res.x.reshape(1,28), X, y))
# theta = res.x


##### Plotting the Results
# fig = plt.figure(figsize=(7,6))
# ax = fig.add_subplot(111)

# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.get_xaxis().tick_bottom()
# ax.get_yaxis().tick_left()

# for i, test in enumerate(y):
# 	if (int(test) == 0):
# 		na = ax.scatter(X[i,1], X[i,2], color="#FDB515", marker="o", alpha=0.5, edgecolor='k')
# 	else:
# 		a = ax.scatter(X[i,1], X[i,2], color="#003262", marker="+", s=75, alpha=0.75, edgecolor='k')

# delta = 0.05
# u = np.arange(-0.8, 1.2, delta)
# v = np.arange(-0.8, 1.2, delta)
# U, V = np.meshgrid(u, v)
# W = np.zeros((len(u),len(v)))
# for i in range(len(u)):
# 	for j in range(len(v)):
# 		W[i,j] = np.dot(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)), theta.T)

# plt.contour(U, V, W.T, levels=[0], cmap=plt.cm.Paired, linestyles="dashed")

# plt.xlim(-0.95, 1.2)
# plt.ylim(-0.8, 1.2)
# plt.title(r"$\lambda=0$")
# plt.xlabel("Microchip Test 1")
# plt.ylabel("Microchip Test 2")
# plt.legend([a, na], ["Pass", "Fail"], loc="upper right", scatterpoints=1, fancybox=False)
# plt.show()
###############