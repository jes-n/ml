import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
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


X = np.loadtxt(os.path.join("..", "data", "ex2data1.csv"), delimiter=",")
### Transform 'X' to follow standard notation of (# features)x(# data points)
X = X.T
### Assign 'y' as the classification vector
y = X[-1]
### Remove 'y' from X
## In NumPy: 'axis=0' corresponds to the rows and 'axis=1' corresponds to the columns
X = np.delete(X, -1, axis=0)
## 'n' is the # of features and 'm' is the # of data points
[n,m] = X.shape

print("X:", X.shape, type(X))
print("y:", y.shape, type(y))

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

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

for i, admission in enumerate(y):
	if (int(admission) == 0):
		na = ax.scatter(X[i,1], X[i,2], color="#FDB515", marker="o", alpha=0.5, edgecolor='k')
	else:
		a = ax.scatter(X[i,1], X[i,2], color="#003262", marker="+", s=75, alpha=0.75, edgecolor='k')

boundary = lambda x, theta: (-1.0/theta[2])*(theta[1]*x + theta[0])
x = np.linspace(29, 101, 50)
plt.plot(x, boundary(x, theta), color="#3B7EA1", alpha=0.75, lw=0.85, ls="--")
plt.xlim(29,101)
plt.ylim(29,101)

plt.xlabel("Exam 1 Score")
plt.ylabel("Exam 2 Score")
plt.legend([na, a], ["Not Admitted", "Admitted"], loc="upper right", scatterpoints=1, fancybox=False)
plt.show()