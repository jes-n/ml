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

def costFunctionReg(theta, X, y, lamb=0):
	hyp = lambda X: sigmoid(np.dot(X,theta.T))

	theta = theta.reshape(1,X.shape[1])
	cost = _costFunctionReg(theta, X, y)
	grad = []
	for i in range(X.shape[1]):
		grad.append(np.average(np.multiply((hyp(X)-y), X[:,[i]])))
	return cost, grad

def mapFeature(X1, X2, degree=6):
	dim = 1
	# print(X1.shape, X2.shape)
	X = np.ones(X1.shape[0]).reshape(X1.shape[0], 1)
	for i in range(1, degree+1):
		for j in range(i+1):
			X = np.column_stack((X, np.multiply(np.power(X1, (i-j)), np.power(X2, j))))
			dim += 1
	return X

X = np.loadtxt(os.path.join("..", "data", "ex2data2.txt"), delimiter=",")
[m,n] = X.shape
y = X[:,[-1]]
X = np.delete(X, -1, axis=1)

### Add a column of ones to X
X = np.column_stack((np.ones(m), X))
# cost, grad = costFunction(theta, X, y)

X = mapFeature(X[:,[1]], X[:,[2]])

theta = np.zeros((1,X.shape[1]))
# cost, grad = costFunctionReg(theta, X, y)
# print("For theta = [0, 0, 0]")
# print("Cost:", cost)
# print("Expected cost (approx): 0.693")
# print("Grad:", grad)
# print("Expected gradients (approx): [-0.1000, -12.0092, -11.2628]")

res = minimize(_costFunctionReg, theta, args=(X, y),
	method='BFGS', options={"maxiter":400}, callback=None)

print("\nSolution:", res.x, "with cost =", _costFunctionReg(res.x.reshape(1,28), X, y))
theta = res.x

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

delta = 0.05
u = np.arange(-0.8, 1.2, delta)
v = np.arange(-0.8, 1.2, delta)
U, V = np.meshgrid(u, v)
W = np.zeros((len(u),len(v)))
for i in range(len(u)):
	for j in range(len(v)):
		# print(mapFeature(u([]), v(j)))
		# print(u[i].reshape(1,1).shape, v[j].shape
		# print(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)))
		# print(np.dot(mapFeature(u[i].reshape(,11), v[j].reshape(1,1)), theta.T))
		W[i,j] = np.dot(mapFeature(u[i].reshape(1,1), v[j].reshape(1,1)), theta.T)

print(W)
plt.contour(U, V, W.T, levels=[0], cmap=plt.cm.Paired, linestyles="dashed")
# plt.colorbar()

plt.xlim(-0.95, 1.2)
plt.ylim(-0.8, 1.2)
plt.title(r"$\lambda=0$")
plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")
plt.legend([na, a], ["y=0", "y=1"], loc="upper right", scatterpoints=1, fancybox=False)
plt.show()