import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize

def sigmoid(z):
	return 1/(1+np.exp(-z))

def _costFunction(theta, X, y):
	_hyp = lambda X: sigmoid(np.dot(X,theta.T))

	theta = theta.reshape(1,X.shape[1])
	return np.average(np.multiply(-y,np.log(_hyp(X))) - np.multiply((1-y),np.log(1-_hyp(X))))

def costFunction(theta, X, y):
	hyp = lambda X: sigmoid(np.dot(X,theta.T))

	theta = theta.reshape(1,X.shape[1])
	cost = _costFunction(theta, X, y)
	grad = []
	for i in range(X.shape[1]):
		grad.append(np.average(np.multiply((hyp(X)-y), X[:,[i]])))
	return cost, grad


X = np.loadtxt(os.path.join("..", "ex2data1.txt"), delimiter=",")
[m,n] = X.shape
y = X[:,[-1]]
X = np.delete(X, -1, axis=1)

### Add a column of ones to X
X = np.column_stack((np.ones(m), X))

theta = np.zeros((1,n))
cost, grad = costFunction(theta, X, y)
print("For theta = [0, 0, 0]")
print("Cost:", cost)
print("Expected cost (approx): 0.693")
print("Grad:", grad)
print("Expected gradients (approx): [-0.1000, -12.0092, -11.2628]")

theta = np.array([[-24, 0.2, 0.2]])
cost, grad = costFunction(theta, X, y)
print("\nFor theta = [-24, 0.2, 0.2]")
print("Cost:", cost)
print("Expected cost (approx): 0.218")
print("Grad:", grad)
print("Expected gradients (approx): [0.043, 2.566, 2.647]")

res = minimize(_costFunction, theta, args=(X, y),
	method='BFGS', options={"maxiter":400}, callback=None)

print("\nSolution:", res.x, "with cost =", _costFunction(res.x.reshape(1,3), X, y))
print("Expected cost (approx):", 0.203)
print("Prediciton:", float(sigmoid(np.dot(np.array([1, 45, 85]), res.x.reshape(1,3).T))))
print("Expected prediction of", 0.776)
theta = res.x

fig = plt.figure()
ax = fig.add_subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

ax.scatter(X[:,1], X[:,2], c=y, cmap=cm.Paired)

boundary = lambda x, theta: (-1.0/theta[2])*(theta[1]*x + theta[0])
x = np.linspace(30, 100, 50)
plt.plot(x, boundary(x, theta))
plt.xlim(30,100)
plt.ylim(30,100)

plt.show()
