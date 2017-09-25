import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize

def sigmoid(z):
	return 1/(1+np.exp(-z))

def _costFunctionReg(theta, X, y):
	_hyp = lambda X: sigmoid(np.dot(X,theta.T))

	theta = theta.reshape(1,X.shape[1])
	return np.average(np.multiply(-y,np.log(_hyp(X))) - np.multiply((1-y),np.log(1-_hyp(X))))

def costFunctionReg(theta, X, y):
	hyp = lambda X: sigmoid(np.dot(X,theta.T))

	theta = theta.reshape(1,X.shape[1])
	cost = _costFunctionReg(theta, X, y)
	grad = []
	for i in range(X.shape[1]):
		grad.append(np.average(np.multiply((hyp(X)-y), X[:,[i]])))
	return cost, grad

def mapFeature(X1, X2, degree=6):
	dim = 1
	X = np.ones(X1.shape[0]).reshape(X1.shape[0], 1)
	for i in range(1, degree+1):
		for j in range(i+1):
			X = np.column_stack((X, np.multiply(np.power(X1, (i-j)), np.power(X2, j))))
			dim += 1
	return X

X = np.loadtxt("ex2data2.txt", delimiter=",")
[m,n] = X.shape
y = X[:,[-1]]
X = np.delete(X, -1, axis=1)

### Add a column of ones to X
X = np.column_stack((np.ones(m), X))
# cost, grad = costFunction(theta, X, y)

X = mapFeature(X[:,[1]], X[:,[2]])

theta = np.zeros((1,X.shape[1]))
cost, grad = costFunctionReg(theta, X, y)
print("For theta = [0, 0, 0]")
print("Cost:", cost)
print("Expected cost (approx): 0.693")
print("Grad:", grad)
print("Expected gradients (approx): [-0.1000, -12.0092, -11.2628]")

# res = minimize(_costFunctionReg, theta, args=(X, y),
# 	method='BFGS', options={"maxiter":400}, callback=None)

# print("\nSolution:", res.x, "with cost =", _costFunctionReg(res.x.reshape(1,28), X, y))
# # print("Expected cost (approx):", 0.203)
# # print("Prediciton:", float(sigmoid(np.dot(np.array([1, 45, 85]), res.x.reshape(1,28).T))))
# # print("Expected prediction of", 0.776)
# theta = res.x

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

for i, test in enumerate(y):
	if (int(test) == 0):
		na = ax.scatter(X[i,1], X[i,2], color="#FDB515", marker="o", alpha=0.5, edgecolor='k')
	else:
		a = ax.scatter(X[i,1], X[i,2], color="#003262", marker="+", s=75, alpha=0.75, edgecolor='k')

# boundary = lambda x, theta: (-1.0/theta[2])*(theta[1]*x + theta[0])
# x = np.linspace(29, 101, 50)
# plt.plot(x, boundary(x, theta), color="#3B7EA1", alpha=0.75, lw=0.85, ls="--")

plt.xlim(-0.95, 1.2)
plt.ylim(-0.8, 1.2)
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend([na, a], ["y=0", "y=1"], loc="upper right", scatterpoints=1, fancybox=False)
plt.show()